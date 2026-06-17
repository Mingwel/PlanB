import numpy as np
from scipy.optimize import root, root_scalar, least_squares
from scipy.integrate import trapezoid, quad, cumulative_trapezoid, IntegrationWarning
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings

# 保证控制台清洁
warnings.filterwarnings("ignore", category=IntegrationWarning)

class TrackedVehicleModel:
    def __init__(self):
        # ==========================================
        # 1. 车辆与履带基础几何参数
        # ==========================================
        self.W_target = 220.257     # 车辆总重，kN
        self.N = 9                 # 负重轮（相同半径）个数
        self.l_0 = 0.75            # 相邻负重轮中心距，m
        
        self.R_r = 0.275           # 负重轮半径，m 
        
        self.R_idler = 0.315       # 前张紧轮(Idler)半径，m
        self.R_sprocket = 0.355    # 后驱动轮(Sprocket)半径，m
        
        self.R_c = 0.05            # 托带轮半径，m
        
        self.b = 2                 # 履带宽度，m
        self.T_0 = 16.67           # 计算履带初始长度所用张力，kN
        self.w_track = 2.08        # 单位长度履带重量，kN/m
        self.t_track = 0.096       # 履带板厚度，m

        # 由于履带板存在厚度，等效计算半径
        # 后续所有计算，均使用_eff等效半径
        self.R_r_eff = self.R_r + self.t_track
        self.R_idler_eff = self.R_idler + self.t_track        
        self.R_sprocket_eff = self.R_sprocket + self.t_track  
        self.R_c_eff = self.R_c + self.t_track

        # 局部坐标系：履带平放，以第一个负重轮中心为原点，向右为x轴，垂直向下为z轴
        # 托带轮的局部坐标，m
        self.carrier_rollers_local = [(1.8, -0.96), (4.5, -0.96)]
        # 前张紧轮相对第一个负重轮的局部坐标偏移量，m
        self.dx_f, self.dy_f = 0.8, 0.62
        # 后驱动轮相对最后一个负重轮的局部坐标偏移量，m
        self.dx_r, self.dy_r = 0.7, 0.6
        # 张紧弹簧柔度系数，m/kN
        self.c_spring = 0.0018
        # 车辆重心的局部坐标，m
        self.x_cg_local = 2.94
        self.z_cg_local = -0.3
        # 牵引点的局部坐标，m
        self.x_p_local = 6.53
        self.z_p_local = -0.06
        
        # ==========================================
        # 2. Bekker沉陷模型与 JH 剪切参数
        # ==========================================
        self.k = 20.01             # kN/m^(n+2)，即Bekker沉陷模量k_c/b + k_phi
        self.n = 0.56              # 无量纲，Bekker沉陷指数
        self.k_0 = 0               # 卸载模量常数项 ，kN/m3
        self.A_u = 1000            # 卸载模量随历史最大下陷深度的增长率 ，kN/m4
        self.ratio_zmax = 0.95     # 尾轮离开土壤的深度判据
        
        self.h = 0                 # 履齿高度，m
        self.gamma = 15.0          # 土壤重度，kN/m3
        self.c = 3                 # 土壤内聚力，kPa
        self.phi = np.deg2rad(2)   # 土壤内摩擦角，转为rad
        self.K = 0.031             # JH模型剪切变形模量，m
        self.slip = 0.05            # 打滑率

        # =========================================================================
        # 3. 内部状态变量初始化
        # =========================================================================
        self.iter_count = 0        # 求解器迭代次数追踪
        self.L_initial_total = 0.0 # 履带刚性静止于平地时的基准总长度，m
        self.F_traction = 0.0      # 总牵引力，kN
        self.F_dp = 0.0            # 净牵引力，kN
        self.best_data = None      # 记录最后一次成功收敛的数据和图形
        
        # 调用函数，计算初始履带总长
        self._init_baseline_length()

    def to_global(self, x_local, z_local, delta, z_u1_out):
        """
        局部坐标系向全局坐标系的转换函数。
        delta: 车辆俯仰角，rad
        z_u1_out: 第一个负重轮下方【履带最外侧边缘】陷入地下的绝对深度，m
        """
        # 利用旋转矩阵，同时扣除首轮等效半径带来的高度差
        x_g = x_local * np.cos(delta) - z_local * np.sin(delta) + self.R_r_eff * np.sin(delta)
        z_g = x_local * np.sin(delta) + z_local * np.cos(delta) + z_u1_out - self.R_r_eff * np.cos(delta)
        return x_g, z_g

    def _solve_single_catenary(self, X1, Z1, R1, X2, Z2, R2, a_cat, init_th1, init_th2):
        """
        求解任意两个圆(轮子)之间公切悬链线的方法，用于求解上方履带段的悬链线形状。
        核心原理：构建包含 4 个未知数(悬链线理论最低点 x_m, z_m 以及两个轮子上的脱离角 th1, th2)的非线性方程组。
        """
        def cat_system(vars):
            x_m, z_m, th1, th2 = vars
            # 圆上切点的坐标
            xA, zA = X1 + R1 * np.cos(th1), Z1 + R1 * np.sin(th1)
            xZ, zZ = X2 + R2 * np.cos(th2), Z2 + R2 * np.sin(th2)
            
            # 条件1,2: 切点必须满足悬链线方程 z = z_m - a * cosh((x - x_m)/a)
            # 局部坐标Z轴向下为正，生成向下悬垂的凹陷曲线，因此用减号
            eq1 = zA - (z_m - a_cat * np.cosh((xA - x_m) / a_cat))
            eq2 = zZ - (z_m - a_cat * np.cosh((xZ - x_m) / a_cat))
            
            # 条件3,4: 圆在切点处的切线斜率必须等于悬链线在切点处的斜率
            eq3 = -np.cos(th1)/np.sin(th1) + np.sinh((xA - x_m) / a_cat)
            eq4 = -np.cos(th2)/np.sin(th2) + np.sinh((xZ - x_m) / a_cat)
            return [eq1, eq2, eq3, eq4]
            
        # 给求解器提供一个初始猜测值 (这里 z_m 是数学方程里的极值准线，不是物理实际最低点)
        x_m_guess = (X1 + X2) / 2.0
        z_m_guess = min(Z1 - R1, Z2 - R2) + a_cat 
        
        sol = root(cat_system, [x_m_guess, z_m_guess, init_th1, init_th2], method='lm')
        x_m, z_m, th1, th2 = sol.x
        
        # 计算该段悬链线的准确弧长积分解析解
        xA = X1 + R1 * np.cos(th1)
        xZ = X2 + R2 * np.cos(th2)
        L_cat = a_cat * (np.sinh((xZ - x_m)/a_cat) - np.sinh((xA - x_m)/a_cat))
        return x_m, z_m, th1 % (2*np.pi), th2 % (2*np.pi), L_cat

    def calc_upper_track_geometry(self, T, delta, z_u1_out):
        """
        计算车辆【顶部悬空履带】的几何形状和总弧长
        涵盖：前张紧轮 -> 托带轮1 -> 托带轮2 -> 后驱动轮
        """
        # 弹簧的压缩导致前轮在局部坐标系x轴上的局部偏移量
        dx_spring = self.c_spring * (T - self.T_0)

        # 1. 计算各个轮子在全局坐标系下的圆心坐标 (注意使用等效外径为基准)
        # 前轮(张紧轮 Idler)：张力大于T0时弹簧压缩，张紧轮向车尾(+X方向)退缩
        X_tc_g, Z_tc_g = self.to_global(-self.dx_f + dx_spring, -self.dy_f, delta, z_u1_out)
        # 后轮(驱动轮 Sprocket)：相对底盘刚性固定
        X_sc_g, Z_sc_g = self.to_global((self.N - 1) * self.l_0 + self.dx_r, -self.dy_r, delta, z_u1_out)
        # 第一个负重轮
        X_c0, Z_c0 = self.to_global(0, 0, delta, z_u1_out)                                      
        # 最后一个负重轮
        X_c_last, Z_c_last = self.to_global((self.N - 1) * self.l_0, 0, delta, z_u1_out)
        # 托带轮1
        X_cr1, Z_cr1 = self.to_global(self.carrier_rollers_local[0][0], self.carrier_rollers_local[0][1], delta, z_u1_out) # 托带轮1
        # 托带轮2
        X_cr2, Z_cr2 = self.to_global(self.carrier_rollers_local[1][0], self.carrier_rollers_local[1][1], delta, z_u1_out) # 托带轮2

        # 2. 计算前轮和后轮的脱离角（相对于水平线的角度），以确定悬链线的边界条件
        df = np.hypot(X_c0 - X_tc_g, Z_c0 - Z_tc_g)
        gf = np.arctan2(Z_c0 - Z_tc_g, X_c0 - X_tc_g)
        alpha_A = gf + np.arccos((self.R_idler_eff - self.R_r_eff) / df) 

        dr = np.hypot(X_c_last - X_sc_g, Z_c_last - Z_sc_g)
        gr = np.arctan2(Z_c_last - Z_sc_g, X_c_last - X_sc_g)
        alpha_Z = gr - np.arccos((self.R_sprocket_eff - self.R_r_eff) / dr)
        
        # 3. 计算悬链线的参数 a_cat
        a_cat = T / self.w_track
        
        # 4. 封装所有的顶部轮子信息
        wheels = [
            {'X': X_tc_g, 'Z': Z_tc_g, 'R': self.R_idler_eff},     # 前张紧轮
            {'X': X_cr1,  'Z': Z_cr1,  'R': self.R_c_eff},         # 托带轮1
            {'X': X_cr2,  'Z': Z_cr2,  'R': self.R_c_eff},         # 托带轮2
            {'X': X_sc_g, 'Z': Z_sc_g, 'R': self.R_sprocket_eff}   # 后驱动轮
        ]
        
        # 分段求解顶部3段小悬链线的解析参数
        xm0, zm0, th_out0, th_in1, L_cat0 = self._solve_single_catenary(wheels[0]['X'], wheels[0]['Z'], wheels[0]['R'], wheels[1]['X'], wheels[1]['Z'], wheels[1]['R'], a_cat, -np.pi/4, -3*np.pi/4)
        xm1, zm1, th_out1, th_in2, L_cat1 = self._solve_single_catenary(wheels[1]['X'], wheels[1]['Z'], wheels[1]['R'], wheels[2]['X'], wheels[2]['Z'], wheels[2]['R'], a_cat, -np.pi/4, -3*np.pi/4)
        xm2, zm2, th_out2, th_in3, L_cat2 = self._solve_single_catenary(wheels[2]['X'], wheels[2]['Z'], wheels[2]['R'], wheels[3]['X'], wheels[3]['Z'], wheels[3]['R'], a_cat, -np.pi/4, -3*np.pi/4)
        
        # 5. 计算各个轮子上的履带包络（缠绕）弧长
        L_wrap_tc = wheels[0]['R'] * ((th_out0 - alpha_A) % (2*np.pi))
        L_wrap_c1 = wheels[1]['R'] * ((th_out1 - th_in1) % (2*np.pi))
        L_wrap_c2 = wheels[2]['R'] * ((th_out2 - th_in2) % (2*np.pi))
        L_wrap_sc = wheels[3]['R'] * ((alpha_Z - th_in3) % (2*np.pi))
        
        # 6. 汇总几何信息，返回总长度和一个包含关键几何参数的字典，供后续计算和可视化使用
        geom = {
            'X_tc': X_tc_g, 'Z_tc': Z_tc_g, 'X_sc': X_sc_g, 'Z_sc': Z_sc_g,
            'alpha_A': alpha_A, 'alpha_Z': alpha_Z, 
            'th_A2': th_out0, 'th_Z2': th_in3, 
            'X_tf': X_tc_g + self.R_idler_eff * np.cos(alpha_A), 'Z_tf': Z_tc_g + self.R_idler_eff * np.sin(alpha_A),
            'X_tr': X_sc_g + self.R_sprocket_eff * np.cos(alpha_Z), 'Z_tr': Z_sc_g + self.R_sprocket_eff * np.sin(alpha_Z),
            'carriers': [
                {'X': X_cr1, 'Z': Z_cr1, 'R': self.R_c_eff, 'th_in': th_in1, 'th_out': th_out1},
                {'X': X_cr2, 'Z': Z_cr2, 'R': self.R_c_eff, 'th_in': th_in2, 'th_out': th_out2}
            ],
            'cats': [
                {'x_m': xm0, 'z_m': zm0, 'a_cat': a_cat, 'th1': th_out0, 'th2': th_in1, 'W1': wheels[0], 'W2': wheels[1]},
                {'x_m': xm1, 'z_m': zm1, 'a_cat': a_cat, 'th1': th_out1, 'th2': th_in2, 'W1': wheels[1], 'W2': wheels[2]},
                {'x_m': xm2, 'z_m': zm2, 'a_cat': a_cat, 'th1': th_out2, 'th2': th_in3, 'W1': wheels[2], 'W2': wheels[3]}
            ]
        }
        # 上半部分总长度，也即A1Z1段的长度
        total_upper_L = L_wrap_tc + L_cat0 + L_wrap_c1 + L_cat1 + L_wrap_c2 + L_cat2 + L_wrap_sc

        return total_upper_L, geom

    def _init_baseline_length(self):
        """ 
        初始化：在平坦坚硬地面上，且使用初始张力T0时，车辆履带的理论总长度，作为长度守恒的基准值。
        """
        L_up_top, geom = self.calc_upper_track_geometry(self.T_0, 0.0, 0.0)
        X_A1, Z_A1 = geom['X_tf'], geom['Z_tf']
        X_Z1, Z_Z1 = geom['X_tr'], geom['Z_tr']
        
        # 首个和最后一个负重轮切点的全局坐标
        X_A = self.to_global(0,0,0,0)[0] + self.R_r_eff * np.cos(geom['alpha_A'])
        Z_A = -self.R_r_eff + self.R_r_eff * np.sin(geom['alpha_A'])
        X_Z = (self.N - 1) * self.l_0 + self.R_r_eff * np.cos(geom['alpha_Z'])
        Z_Z = -self.R_r_eff + self.R_r_eff * np.sin(geom['alpha_Z'])
        
        # 勾股定理，计算前张紧轮-首轮、后驱动轮-尾轮公切线长度
        L_front_tan = np.hypot(X_A - X_A1, Z_A - Z_A1)
        L_rear_tan = np.hypot(X_Z1 - X_Z, Z_Z1 - Z_Z)
        
        # 底部平地绕行段长度
        L_bot = self.R_r_eff * (geom['alpha_A'] - np.pi/2) + (self.N - 1) * self.l_0 + self.R_r_eff * (np.pi/2 - geom['alpha_Z'] % (2*np.pi))
        
        self.L_initial_total = L_up_top + L_front_tan + L_rear_tan + L_bot
        print(f"\n{'='*60}\n✅ 初始总长定标: {self.L_initial_total:.6f} m\n{'='*60}\n")

    def solve_segment(self, i, T, delta, z_u1_out):
        """
        【理论解析核心】：求解两个负重轮之间，仅与软土接触的履带段形状。
        这是一个隐式积分寻根过程，通过猜测脱离角 phi_ri 来让水平长度守恒。
        """
        # 将总张力 (kN) 转化为单位宽度张力 (kN/m)，供内部公式积分使用
        T_unit = T / self.b 
        
        # 计算第 i 个 和第 i+1 个负重轮中心在全局坐标系中的深度
        z_ui = z_u1_out + i * self.l_0 * np.sin(delta)
        z_ui_next = z_u1_out + (i + 1) * self.l_0 * np.sin(delta)
        z_ci_next = z_ui_next - self.R_r_eff
        
        # 计算某个轮下的k_ui、z_ai
        k_ui = self.k_0 + self.A_u * max(z_ui, 0)
        p_ui = self.k * (z_ui ** self.n) if z_ui > 0 else 0
        z_ai = z_ui - p_ui / k_ui if k_ui != 0 else z_ui
        
        stat = {}

        def horizontal_error(phi_ri):
            # z_ri: 履带与负重轮分离点的深度
            z_ri = z_ui - self.R_r_eff * (1 - np.cos(phi_ri))
            # C_1i: 积分常数，代表该段履带内部维持平衡的几何常量
            C_1i = -T_unit * np.cos(phi_ri) - k_ui * z_ri * (z_ri / 2.0 - z_ai)

            # z_mi: 该段履带在两轮之间能达到的最小深度（凹陷最高点）
            val_under = (z_ai - z_ri)**2 - (2 * T_unit / k_ui) * (1 - np.cos(phi_ri))
            if val_under < 0: return 1e5
            z_mi = z_ai + np.sqrt(val_under)

            # 预测履带与下一个负重轮接触时的深度
            term1 = T_unit / (2 * self.R_r_eff * k_ui) - z_ai + self.R_r_eff * np.cos(phi_ri) + z_ci_next
            val2_under = (z_ai - z_ri)**2 + (2 * T_unit / (self.R_r_eff * k_ui)) * term1
            if val2_under < 0: return 1e5
            z_li_next_hyp1 = z_ai - T_unit / (self.R_r_eff * k_ui) + np.sqrt(val2_under)
            
            # --- 卸载区积分方程 (Unloading) ---
            def int_unload(z):
                val = (-T_unit / (k_ui * z * (z / 2.0 - z_ai) + C_1i))**2 - 1.0
                return 1.0 / np.sqrt(max(val, 1e-10))

            # Case 1: 接触下一个轮子时，尚未进入新一轮土壤加载区
            if z_li_next_hyp1 <= z_ui:
                I1 = abs(quad(int_unload, z_mi, z_ri, points=[z_mi])[0])
                I2 = abs(quad(int_unload, z_mi, z_li_next_hyp1, points=[z_mi])[0])
                phi_li_next = np.arccos(np.clip((z_li_next_hyp1 - z_ci_next)/self.R_r_eff, -1.0, 1.0))
                L_x = self.R_r_eff * (np.sin(phi_ri) + np.sin(phi_li_next)) + I1 + I2
                stat.update({'case':1, 'z_ui':z_ui, 'z_ri':z_ri, 'z_mi':z_mi, 'z_li_next':z_li_next_hyp1, 'phi_li_next':phi_li_next, 'int_unload':int_unload})
            
            # Case 2: 接触下一个轮子前，深度突破了历史纪录，进入再加载区 (Loading)
            else:
                S_E_inner = k_ui * z_ui * (z_ui / 2.0 - z_ai) + C_1i

                # 加载区能量守恒隐式方程，基于 Bekker 模型 p = k*z^n 的完美数学解析积分解
                def eq30(z_l):
                    integral_p = (self.k / (self.n + 1)) * (z_l**(self.n + 1) - z_ui**(self.n + 1))
                    return self.R_r_eff * integral_p + self.R_r_eff * S_E_inner + T_unit * (z_l - z_ci_next)
                
                bracket_max = z_ui_next + self.R_r_eff
                if np.sign(eq30(z_ui)) == np.sign(eq30(bracket_max)): return 1e5
                z_li_next = root_scalar(eq30, bracket=[z_ui, bracket_max], method='brentq').root
                phi_li_next = np.arccos(np.clip((z_li_next - z_ci_next)/self.R_r_eff, -1.0, 1.0))
                
                # 加载区积分方程
                def int_load(z):
                    integral_p = (self.k / (self.n + 1)) * (z**(self.n + 1) - z_ui**(self.n + 1))
                    denom_inner = integral_p + S_E_inner
                    val = (-T_unit / denom_inner)**2 - 1.0
                    return 1.0 / np.sqrt(max(val, 1e-10))
                
                I1, I2 = abs(quad(int_unload, z_mi, z_ri, points=[z_mi])[0]), abs(quad(int_unload, z_mi, z_ui, points=[z_mi])[0])
                I3 = abs(quad(int_load, z_ui, z_li_next)[0])
                
                L_x = self.R_r_eff * (np.sin(phi_ri) + np.sin(phi_li_next)) + I1 + I2 + I3
                stat.update({'case':2, 'z_ui':z_ui, 'z_ri':z_ri, 'z_mi':z_mi, 'z_li_next':z_li_next, 'phi_li_next':phi_li_next, 'int_unload':int_unload, 'int_load':int_load})

            # 返回水平跨距误差（理论跨度减去车身轮间距的水平投影）
            return L_x - self.l_0 * np.cos(delta)

        # 遍历试凑分离角 phi_ri 寻找过零点
        search = np.linspace(1e-4, np.pi - 1e-3, 30)
        for j in range(len(search)-1):
            if np.sign(horizontal_error(search[j])) != np.sign(horizontal_error(search[j+1])):
                stat['phi_ri'] = root_scalar(horizontal_error, bracket=[search[j], search[j+1]], method='brentq').root
                return stat
        raise ValueError("Segment Solve Failed")

    def assemble_full_track(self, T_top, delta, z_u1_out, inner_results):
        """ 将解算出的各个独立软土履带段与包络段拼接，生成完整的 (X, Z) 离散点集用于受力评估和画图 """
        X_full, Z_full = [], []
        kp = {'A':[], 'A1':[], 'A2':[], 'Z':[], 'Z1':[], 'Z2':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'X':[], 'V':[]}
        L_up_top, ug = self.calc_upper_track_geometry(T_top, delta, z_u1_out)
        
        # 拼接首部直线段
        X_A1, Z_A1 = ug['X_tf'], ug['Z_tf']
        X_A, Z_A = self.to_global(0,0,delta,z_u1_out)[0] + self.R_r_eff*np.cos(ug['alpha_A']), self.to_global(0,0,delta,z_u1_out)[1] + self.R_r_eff*np.sin(ug['alpha_A'])
        X_full.extend(np.linspace(X_A1, X_A, 20)); Z_full.extend(np.linspace(Z_A1, Z_A, 20))
        
        # 循环拼接各个负重轮及其之间的软土接触段
        for i in range(self.N):
            X_ci, Z_ci = self.to_global(i*self.l_0, 0, delta, z_u1_out)
            kp['B'].append((X_ci, Z_ci + self.R_r_eff))
            
            # 轮子上的履带包角
            ths = np.linspace(ug['alpha_A'] if i==0 else np.pi/2 + inner_results[i-1]['phi_li_next'], 
                              ug['alpha_Z'] if i==self.N-1 else np.pi/2 - inner_results[i]['phi_ri'], 60)
            X_full.extend(X_ci + self.R_r_eff*np.cos(ths)); Z_full.extend(Z_ci + self.R_r_eff*np.sin(ths))

            # 还原悬链线离散点
            if i < self.N - 1:
                stat = inner_results[i]
                kp['C'].append((X_full[-1], Z_full[-1]))
                
                sqrt_z = np.linspace(np.sqrt(stat['z_ri'] - stat['z_mi']), 0, 40)
                z_CD = stat['z_mi'] + sqrt_z**2
                x_CD = X_full[-1] + np.array([abs(quad(stat['int_unload'], stat['z_ri'], z, points=[stat['z_mi']])[0]) for z in z_CD])
                X_full.extend(x_CD[1:]); Z_full.extend(z_CD[1:]); kp['D'].append((x_CD[-1], z_CD[-1]))
                
                if stat['case'] == 1:
                    sqrt_z = np.linspace(0, np.sqrt(stat['z_li_next'] - stat['z_mi']), 40)
                    z_DF = stat['z_mi'] + sqrt_z**2
                    x_DF = x_CD[-1] + np.array([abs(quad(stat['int_unload'], stat['z_mi'], z, points=[stat['z_mi']])[0]) for z in z_DF])
                    X_full.extend(x_DF[1:]); Z_full.extend(z_DF[1:])
                else:
                    sqrt_z = np.linspace(0, np.sqrt(stat['z_ui'] - stat['z_mi']), 30)
                    z_DE = stat['z_mi'] + sqrt_z**2
                    x_DE = x_CD[-1] + np.array([abs(quad(stat['int_unload'], stat['z_mi'], z, points=[stat['z_mi']])[0]) for z in z_DE])
                    X_full.extend(x_DE[1:]); Z_full.extend(z_DE[1:]); kp['E'].append((x_DE[-1], z_DE[-1]))
                    
                    z_EF = np.linspace(stat['z_ui'], stat['z_li_next'], 30)
                    x_EF = x_DE[-1] + np.array([abs(quad(stat['int_load'], stat['z_ui'], z)[0]) for z in z_EF])
                    X_full.extend(x_EF[1:]); Z_full.extend(z_EF[1:])
                kp['F'].append((X_full[-1], Z_full[-1]))

        # 拼接尾部直线段
        X_Z, Z_Z = self.to_global((self.N - 1)*self.l_0,0,delta,z_u1_out)[0] + self.R_r_eff*np.cos(ug['alpha_Z']), self.to_global((self.N - 1)*self.l_0,0,delta,z_u1_out)[1] + self.R_r_eff*np.sin(ug['alpha_Z'])
        X_Z1, Z_Z1 = ug['X_tr'], ug['Z_tr']
        X_full.extend(np.linspace(X_Z, X_Z1, 20)); Z_full.extend(np.linspace(Z_Z, Z_Z1, 20))

        X_full = np.array(X_full); Z_full = np.array(Z_full)

        # 寻找履带与地面的分离点 X
        idx_X = next((k for k in range(len(Z_full)-1) if Z_full[k] <= 0 and Z_full[k+1] > 0), None)
        if idx_X is not None:
            t = (0 - Z_full[idx_X]) / (Z_full[idx_X+1] - Z_full[idx_X])
            X_start = X_full[idx_X] + t * (X_full[idx_X+1] - X_full[idx_X])
        else: X_start = X_full[0]
        kp['X'].append((X_start, 0.0))

        # 寻找履带与地面的最终接触终点 V (基于ratio_zmax)
        X_c_last, Z_c_last = self.to_global((self.N - 1)*self.l_0, 0, delta, z_u1_out)
        Z_B_last = Z_c_last + self.R_r_eff
        Z_V = self.ratio_zmax * Z_B_last
        X_V = X_c_last + np.sqrt(max(0, self.R_r_eff**2 - (Z_V - Z_c_last)**2))
        kp['V'].append((X_V, Z_V))

        kp['A1'].append((X_A1, Z_A1)); kp['Z1'].append((X_Z1, Z_Z1))
        kp['A2'].append((ug['X_tc'] + self.R_idler_eff*np.cos(ug['th_A2']), ug['Z_tc'] + self.R_idler_eff*np.sin(ug['th_A2'])))
        kp['Z2'].append((ug['X_sc'] + self.R_sprocket_eff*np.cos(ug['th_Z2']), ug['Z_sc'] + self.R_sprocket_eff*np.sin(ug['th_Z2'])))
        kp['A'].append((X_A, Z_A)); kp['Z'].append((X_Z, Z_Z))
        
        return X_full, Z_full, kp, L_up_top, ug

    def compute_residuals(self, vars):
        """
        【最优化残差函数】：输入三个猜测变量，输出三大物理约束的误差。
        vars = [T1 (初始首端张力), delta (俯仰角), z_u1_out (首轮外边缘下陷深度)]
        """
        T1, delta, z_u1_out = vars
        if T1 < 0.5 or z_u1_out < 0.001 or delta < -np.pi/4: return [1e5, 1e5, 1e5]
        
        try:
            # =========================================================================
            # Phase 1: 基础静态场初始化 (不考虑剪应力)
            # 给定一个恒定的张力 T1，试算履带的下陷情况和法向压力分布
            # =========================================================================
            results_p1 = [self.solve_segment(i, T1, delta, z_u1_out) for i in range(self.N - 1)]
            X_full, Z_full, kp, L_top1, ug1 = self.assemble_full_track(T1, delta, z_u1_out, results_p1)
            
            X_start, X_V = kp['X'][0][0], kp['V'][0][0]
            mask = (X_full >= X_start) & (X_full <= X_V)
            X_sup, Z_sup = X_full[mask], Z_full[mask]
            
            P_sup = np.zeros_like(Z_sup)
            max_z = 0.0
            for k, (x, z) in enumerate(zip(X_sup, Z_sup)):
                if z <= 0: continue
                max_z = max(max_z, z)
                if z >= max_z - 1e-6: 
                    P_sup[k] = self.k * (z ** self.n)  # 加载
                else:
                    k_ui = self.k_0 + self.A_u * max_z
                    p_max = self.k * (max_z ** self.n)
                    z_ai = max_z - p_max / k_ui
                    P_sup[k] = k_ui * (z - z_ai) if z > z_ai else 0 # 卸载再加载

            # =========================================================================
            # Phase 2: 剪应力场生成 与 张力累加 (Janosi-Hanamoto 模型)
            # =========================================================================
            dL = np.sqrt(np.diff(X_sup)**2 + np.diff(Z_sup)**2)
            # 沿履带的弧长积分
            L_cum = np.insert(np.cumsum(dL), 0, 0)
            
            # 寻找各段履带重复剪切的起点（D点，即两轮之间履带悬挂得最高的地方）
            S_sup = np.zeros_like(P_sup)
            idx_origins = [0]
            for dx, dz in kp['D']:
                if X_sup[0] < dx < X_sup[-1]:
                    idx = np.argmin(np.abs(X_sup - dx))
                    if idx > idx_origins[-1]: idx_origins.append(idx)
            
            term_sin = np.sin(np.pi/4 + self.phi/2)**2
            term_tan_cos = np.tan(np.pi/4 + self.phi/2) * np.cos(np.pi/2 - self.phi)
            
            # 计算摩擦力
            for seg in range(len(idx_origins)):
                start_idx = idx_origins[seg]
                end_idx = idx_origins[seg+1] if seg < len(idx_origins)-1 else len(X_sup)
                for k in range(start_idx, end_idx):
                    l = L_cum[k] - L_cum[start_idx]
                    x = X_sup[k] - X_sup[start_idx]
                    # 计算剪切位移 j (考虑了滑转率 slip)
                    j = max(l - (1 - self.slip) * x, 0.0)
                    # JH 底面剪切模型
                    tau_max = self.c + P_sup[k] * np.tan(self.phi)
                    tau_k = tau_max * (1 - np.exp(-j / self.K))
                    # 履齿侧边剪切模型 (如果输入了真实履齿高度 h)
                    z_m_local = max(Z_sup[k], 0.0) 
                    V_max = 4 * self.c * self.h * term_sin + 2 * self.h * z_m_local * self.gamma * term_tan_cos
                    V_k = V_max * (1 - np.exp(-j / self.K))
                    # V_k为单位长度履带由履齿产生的牵引力（kN/m），与履带宽度无关，为便于计算，除以单位长度宽度 b，才能得到剪应力
                    S_sup[k] = tau_k + V_k / self.b
            
            # 张力从车头累加到车尾
            tension_inc = cumulative_trapezoid(S_sup * self.b, x=L_cum, initial=0)
            T_dist = T1 + tension_inc
            Tu = T_dist[-1]
            # 提取各悬空段两端的平均张力，供重塑形状使用
            T_CF_avg = []
            for i in range(self.N - 1):
                if i < len(kp['C']) and i < len(kp['F']):
                    idx_C = np.argmin(np.abs(X_sup - kp['C'][i][0]))
                    idx_F = np.argmin(np.abs(X_sup - kp['F'][i][0]))
                    T_CF_avg.append((T_dist[idx_C] + T_dist[idx_F]) / 2.0)
                else: T_CF_avg.append(T1)

            # =========================================================================
            # Phase 3: 柔性履带形状动态重塑 (考入局部张力)
            # =========================================================================
            results_p2 = [self.solve_segment(i, T_CF_avg[i], delta, z_u1_out) for i in range(self.N - 1)]
            
            # 后置驱动轮，上方履带是松边，张力传递的是首端的较小张力 T1
            X_full_new, Z_full_new, kp_new, L_top_new, ug_new = self.assemble_full_track(T1, delta, z_u1_out, results_p2)

            # =========================================================================
            # Phase 4: 积分代换计算三大物理约束方程
            # =========================================================================
            X_start_new, X_V_new = kp_new['X'][0][0], kp_new['V'][0][0]
            mask_new = (X_full_new >= X_start_new) & (X_full_new <= X_V_new)
            X_sup_new, Z_sup_new = X_full_new[mask_new], Z_full_new[mask_new]
            # 重新提取终极形状上的法向力
            P_sup_new = np.zeros_like(Z_sup_new)
            max_z = 0.0
            
            for k, (x, z) in enumerate(zip(X_sup_new, Z_sup_new)):
                if z <= 0: continue
                max_z = max(max_z, z)
                if z >= max_z - 1e-6: 
                    P_sup_new[k] = self.k * (z ** self.n)
                else:
                    k_ui = self.k_0 + self.A_u * max_z
                    p_max = self.k * (max_z ** self.n)
                    z_ai = max_z - p_max / k_ui
                    P_sup_new[k] = k_ui * (z - z_ai) if z > z_ai else 0

            # --- 向量微积分几何代换 (用离散差分矩阵完美替代曲线微元 dl 的三角函数投影) ---
            dX_new = np.diff(X_sup_new) # 等价于 dl * cos(theta)
            dZ_new = np.diff(Z_sup_new) # 等价于 dl * sin(theta)
            
            # 计算微元段的平均受力状态
            P_mid = (P_sup_new[:-1] + P_sup_new[1:]) / 2.0
            S_sup_new = np.interp(X_sup_new, X_sup, S_sup)
            T_dist_new = np.interp(X_sup_new, X_sup, T_dist)
            S_mid = (S_sup_new[:-1] + S_sup_new[1:]) / 2.0
            X_mid = (X_sup_new[:-1] + X_sup_new[1:]) / 2.0
            Z_mid = (Z_sup_new[:-1] + Z_sup_new[1:]) / 2.0
            
            xg_cg, zg_cg = self.to_global(self.x_cg_local, self.z_cg_local, delta, z_u1_out)
            Xp_g, Zp_g = self.to_global(self.x_p_local, self.z_p_local, delta, z_u1_out)

            # --- 最终力与力矩清算 ---
            F_traction = 2 * self.b * np.sum(S_mid * dX_new)         # 土壤向前剪切推力
            R_compaction = 2 * self.b * np.sum(P_mid * dZ_new)       # 土壤压实阻力
            F_dp = F_traction - R_compaction                         # 净牵引力

            # 约束 1：竖直载荷必须与车重抵消
            W_calc = 2 * self.b * np.sum(P_mid * dX_new + S_mid * dZ_new)
            err_W = W_calc - self.W_target
            
            # 约束 2：俯仰力矩必须平衡
            M_P_vert = P_mid * dX_new * (xg_cg - X_mid)        
            M_P_horz = -P_mid * dZ_new * (Z_mid - zg_cg)      
            M_S_horz = S_mid * dX_new * (Z_mid - zg_cg)        
            M_S_vert = S_mid * dZ_new * (xg_cg - X_mid)        
            M_Drawbar = -F_dp * (Zp_g - zg_cg)                
            M_total = 2 * self.b * np.sum(M_P_vert + M_P_horz + M_S_horz + M_S_vert) + M_Drawbar
            
            # 约束 3：整条履带长度必须守恒
            L_bot_new = np.sum(np.sqrt(np.diff(X_full_new)**2 + np.diff(Z_full_new)**2))
            err_L = (L_bot_new + L_top_new) - self.L_initial_total
            
            self.F_traction = F_traction
            self.F_dp = F_dp
        
        # 遇到无解物理几何边界时，抛出极大残差，迫使优化器转向
        except Exception as e:
            return [1e5, 1e5, 1e5]

        self.iter_count += 1
        # 打印迭代信息
        if self.iter_count % 1 == 0:
            print(f"Iter {self.iter_count:>2} | T1={T1:>5.2f} kN, Tu={Tu:>5.2f} kN, δ={np.rad2deg(delta):>5.2f}°, 外缘下陷zu1_out={z_u1_out*100:>5.2f} cm")
            print(f"       -> F_dp={F_dp:>5.2f} kN, errW={err_W:>6.3f} kN, errM={M_total:>6.3f} kN·m, errL={err_L*1000:>6.3f} mm")
        self.best_data = (X_full_new, Z_full_new, X_sup_new, Z_sup_new, P_sup_new, S_sup_new, T_dist_new, kp_new, ug_new, T1, Tu, delta, z_u1_out)
        # 将三项误差按比例放大后返回，供 Least_Squares 计算雅可比梯度并向下寻优
        return [err_W, M_total, err_L * 1000.0]

    def solve(self):
        """ 通过最小二乘法，搜索能让 3 项误差同时降为 0 的最优物理状态变量 """
        res = least_squares(self.compute_residuals, 
                            [15.0, np.deg2rad(1.0), 0.1], 
                            bounds=([0.5, -np.deg2rad(10), 0.01], [200, 0.3, 0.8]), 
                            diff_step=1e-2, ftol=1e-10, xtol=1e-10)
                            
        if res.success and self.best_data is not None:
            # =====================================================================
            # 【底层修补】：防止解算器最后一次算梯度(扰动值)覆盖最佳解，强制用完美解再刷一次
            # =====================================================================
            self.compute_residuals(res.x)
            
            X_full, Z_full, X_sup, Z_sup, P_sup, S_sup, T_dist, kp, ug, final_T1, final_Tu, final_delta, final_zu1 = self.best_data
            # 最终结果打印
            print("\n" + "="*60 + f"\n模型收敛：驱动轮后置、等效物理厚度 ({self.t_track*1000}mm)\n" + "="*60)
            print(f"🎯 总牵引力 (Gross Tractive Effort): {self.F_traction:.3f} kN")
            print(f"🎯 净牵引力 (Drawbar Pull @ P): {self.F_dp:.3f} kN")
            print(f"🎯 最终倾角 (Pitch Angle): {np.rad2deg(final_delta):.3f}°")
            print(f"🎯 首端初张力 (T1): {final_T1:.2f} kN  --> 尾端满张力 (Tu): {final_Tu:.2f} kN")
            
            self.plot_results()
        else:
            print("\n" + "!"*60)
            print("❌ 求解失败：物理模型未能收敛。")
            print("!"*60)

    def plot_results(self):
        """ 绘制学术级高精度车辆地面力学可视化图表 """
        X_full, Z_full, X_sup, Z_sup, P_sup, S_sup, T_dist, kp, ug, T1, Tu, delta, z_u1_out = self.best_data
        output_path = r"D:\理论预测竖向压力_完整版.csv"

        # =========================================================================
        # 将履带法向压力 P_sup 转换为竖向压力分布 P_vertical
        # 原理：P_sup 为垂直于履带板/接触曲线的法向压力标量；
        #       若局部履带切线与水平面的夹角为 alpha，则竖向压力分量为 P_sup*cos(alpha)。
        #       这里不用角度反三角函数，而是直接由离散几何差分获得：
        #       cos(alpha) = |dX| / sqrt(dX^2 + dZ^2)
        #
        # X_sup, Z_sup 是沿履带路径顺序排列的几何点，不一定是严格单值的 X-P 曲线。
        #       在负重轮圆弧、分段拼接点附近，可能出现 dX 接近 0 或同一 X 位置对应多个压力点。
        #       如果直接 ax2.plot(X_pressure, P_vertical)，Matplotlib 会把这些点强行连成竖线。
        #       因此这里先构造微元中点压力，再按水平 X 小区间合并成单值分布。
        # =========================================================================
        dX_sup = np.diff(X_sup)
        dZ_sup = np.diff(Z_sup)
        dL_sup = np.sqrt(dX_sup**2 + dZ_sup**2)

        # 防止重复点或极短线段导致除零；同时剔除近竖直微元。
        # 近竖直微元的水平跨度几乎为零，不适合作为 X-压力曲线上的有限区间；
        # 若保留它们，容易在负重轮中心或几何拼接点形成竖直连线。
        eps_len = 1e-12
        cos_alpha_min = 0.05
        cos_alpha_raw = np.ones_like(dL_sup)
        valid_len = dL_sup > eps_len
        cos_alpha_raw[valid_len] = np.abs(dX_sup[valid_len]) / dL_sup[valid_len]
        valid_seg = valid_len & (cos_alpha_raw >= cos_alpha_min)

        # 将点值压力、张力转换到微元中点，与 cos(alpha) 的定义位置保持一致
        X_pressure_raw = (X_sup[:-1] + X_sup[1:]) / 2.0
        P_mid_export_raw = (P_sup[:-1] + P_sup[1:]) / 2.0
        T_mid_export_raw = (T_dist[:-1] + T_dist[1:]) / 2.0
        P_vertical_raw = P_mid_export_raw * cos_alpha_raw

        # 只保留有效微元；同时去除非接触或无效压力点
        valid_pressure = valid_seg & np.isfinite(X_pressure_raw) & np.isfinite(P_vertical_raw) & (P_vertical_raw > 0.0)
        X_raw = X_pressure_raw[valid_pressure]
        P_raw = P_vertical_raw[valid_pressure]
        T_raw = T_mid_export_raw[valid_pressure]
        C_raw = cos_alpha_raw[valid_pressure]

        # 按 X 位置排序，并将同一水平位置附近的多个点合并。
        # 对每个 X 小区间取竖向压力包络值，避免一个 X 位置被多个履带路径点反复连接成竖线。
        if len(X_raw) > 0:
            order = np.argsort(X_raw)
            X_raw = X_raw[order]
            P_raw = P_raw[order]
            T_raw = T_raw[order]
            C_raw = C_raw[order]

            dx_bin = max(0.002, (X_raw[-1] - X_raw[0]) / 1200.0)
            bin_id = np.floor((X_raw - X_raw[0]) / dx_bin).astype(int)
            unique_bins = np.unique(bin_id)

            X_list, P_list, T_list, C_list = [], [], [], []
            for bid in unique_bins:
                idxs = np.where(bin_id == bid)[0]
                if len(idxs) == 0:
                    continue
                # 取该水平小区间内的竖向压力最大值作为包络；
                # 这样可以避免多值 X 被平均抹平，也能保留负重轮下方峰值。
                pick = idxs[np.argmax(P_raw[idxs])]
                X_list.append(np.mean(X_raw[idxs]))
                P_list.append(P_raw[pick])
                T_list.append(T_raw[pick])
                C_list.append(C_raw[pick])

            X_pressure = np.array(X_list)
            P_vertical = np.array(P_list)
            T_mid_export = np.array(T_list)
            cos_alpha = np.array(C_list)
        else:
            X_pressure = np.array([])
            P_vertical = np.array([])
            T_mid_export = np.array([])
            cos_alpha = np.array([])
            dx_bin = 0.002

        min_len = min(len(X_pressure), len(P_vertical), len(T_mid_export), len(cos_alpha))
        np.savetxt(output_path,
            np.column_stack((X_pressure[:min_len], P_vertical[:min_len], T_mid_export[:min_len], cos_alpha[:min_len])),
            delimiter=",",
            header="X_support_mid (m),Vertical Pressure_kPa,Tension_kN,Cos_alpha",
            comments="",
            fmt="%.6f")
        print(f"✅ 竖向压力与张力数据已保存至：{output_path}")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16), sharex=True)
        # 地表基准面
        ax1.axhline(0, color='brown', ls='--', alpha=0.5, label='Undisturbed Ground (Z=0)')
        
        pts = np.array([X_full, Z_full]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = []
        max_z = 0.0
        X_start, X_V = kp['X'][0][0], kp['V'][0][0]
        
        # 红色代表加载段，蓝色代表卸载再加载段
        for i in range(len(Z_full)-1):
            x_m, z_m = (X_full[i]+X_full[i+1])/2, (Z_full[i]+Z_full[i+1])/2
            max_z = max(max_z, z_m)
            if x_m < X_start or x_m > X_V or z_m <= 0: cols.append('black')
            elif z_m >= max_z - 1e-5: cols.append('red')
            else: cols.append('blue')
            
        # 1. 绘制履带的最外侧轨迹
        ax1.add_collection(LineCollection(segs, colors=cols, linewidths=2.5, alpha=0.9))
        ax1.plot([], [], 'r-', lw=3, label='Loading State (Bekker)')
        ax1.plot([], [], 'b-', lw=3, label='Unloading / Reloading')

        # 2. 绘制真实之轮
        # 负重轮
        for i in range(self.N): 
            X_c_wheel, Z_c_wheel = self.to_global(i*self.l_0, 0, delta, z_u1_out)
            ax1.add_patch(plt.Circle((X_c_wheel, Z_c_wheel), self.R_r, fill=False, color='royalblue', lw=2))
        # 前诱导轮(Idler) 与 后驱动轮(Sprocket)
        X_s_front, Z_s_front = ug['X_tc'], ug['Z_tc']
        X_s_rear, Z_s_rear = ug['X_sc'], ug['Z_sc']
        ax1.add_patch(plt.Circle((X_s_front, Z_s_front), self.R_idler, fill=False, ls='-', color='crimson', lw=2, label='Idler (Front)'))
        ax1.add_patch(plt.Circle((X_s_rear, Z_s_rear), self.R_sprocket, fill=False, ls='-', color='darkred', lw=2, label='Sprocket (Rear)'))
        # 托带轮(Carriers)
        for c in ug['carriers']:
            ax1.add_patch(plt.Circle((c['X'], c['Z']), self.R_c, fill=False, ls='-', color='purple', lw=2))
        # 顶部悬空段与包角
        for cat in ug['cats']:
            x1 = cat['W1']['X'] + cat['W1']['R'] * np.cos(cat['th1'])
            x2 = cat['W2']['X'] + cat['W2']['R'] * np.cos(cat['th2'])
            cx = np.linspace(x1, x2, 100)
            ax1.plot(cx, cat['z_m'] - cat['a_cat'] * np.cosh((cx - cat['x_m']) / cat['a_cat']), 'k-', lw=2.5)

        th_start_A, th_end_A = ug['alpha_A'], ug['th_A2']
        if th_end_A < th_start_A: th_end_A += 2*np.pi
        th_wrap_A = np.linspace(th_start_A, th_end_A, 50)
        ax1.plot(ug['X_tc'] + self.R_idler_eff * np.cos(th_wrap_A), ug['Z_tc'] + self.R_idler_eff * np.sin(th_wrap_A), 'k-', lw=2.5)
        
        th_start_Z, th_end_Z = ug['th_Z2'], ug['alpha_Z']
        if th_end_Z < th_start_Z: th_end_Z += 2*np.pi
        th_wrap_Z = np.linspace(th_start_Z, th_end_Z, 50)
        ax1.plot(ug['X_sc'] + self.R_sprocket_eff * np.cos(th_wrap_Z), ug['Z_sc'] + self.R_sprocket_eff * np.sin(th_wrap_Z), 'k-', lw=2.5)

        # 3. 关键点高亮与图面文本标注
        cmaps = {'A':'k', 'A1':'m', 'A2':'m', 'Z':'k', 'Z1':'m', 'Z2':'m', 'B':'g', 'C':'orange', 'D':'brown', 'E':'purple', 'F':'cyan', 'X':'r', 'V':'r'}
        markers = {'A':'*', 'A1':'X', 'A2':'X', 'Z':'*', 'Z1':'X', 'Z2':'X', 'B':'o', 'C':'^', 'D':'v', 'E':'D', 'F':'s', 'X':'P', 'V':'P'}

        for name, points in kp.items():
            if points:
                xs, zs = zip(*points)
                ax1.scatter(xs, zs, c=[cmaps[name]]*len(xs), marker=markers[name], s=40, label=f"Point {name}", zorder=6)
                for idx, (x, z) in enumerate(points):
                    label_text = f"{name}{idx+1}" if len(points) > 1 and name in ['B', 'C', 'D', 'E', 'F'] else name
                    ax1.annotate(label_text, (x, z), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9, color=cmaps[name], fontweight='bold')
        # 车辆重心 CG 点
        xg_cg, zg_cg = self.to_global(self.x_cg_local, self.z_cg_local, delta, z_u1_out)
        ax1.scatter(xg_cg, zg_cg, color='gold', marker='X', s=60, label='Center of Gravity', edgecolors='k', zorder=7)
        ax1.annotate('CG', (xg_cg, zg_cg), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=10, color='k', fontweight='bold')
        # 净牵引力 P 点
        Xp_g, Zp_g = self.to_global(self.x_p_local, self.z_p_local, delta, z_u1_out)
        ax1.scatter(Xp_g, Zp_g, color='red', marker='s', s=50, label='Drawbar Pull Point P', edgecolors='k', zorder=7)
        ax1.annotate('P', (Xp_g, Zp_g), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=10, color='darkred', fontweight='bold')

        ax1.invert_yaxis(); ax1.set_aspect('equal'); ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax1.set_title(f"Dynamic State Geometry (Rear Sprocket Drive, Track Thickness {self.t_track*1000} mm)", fontsize=14)
        # === 竖向压力图 ===
        # 注意：这里绘制的是 P_sup*cos(alpha)，即履带法向压力在竖直方向上的分量。
        # 使用按 X 合并后的单值分布，避免同一 X 位置的多值点被连成竖线。
        ax2.fill_between(X_pressure[:min_len], P_vertical[:min_len], 0, color='royalblue', alpha=0.3)
        ax2.plot(X_pressure[:min_len], P_vertical[:min_len], 'b-', lw=2)
        ax2.invert_yaxis()
        ax2.set_ylabel("Vertical Pressure $P\\cos\\alpha$ (kPa)", fontsize=12)
        ax2.set_title(f"Vertical Pressure Distribution", fontsize=14)
        ax2.grid(True, linestyle=':', alpha=0.7)
        # === 剪应力与张力双坐标轴图 ===
        min_len = min(len(X_sup), len(S_sup))
        ax3.fill_between(X_sup[:min_len], S_sup[:min_len], 0, color='forestgreen', alpha=0.3, label='Equivalent Total Shear Stress $\\tau_{eq}$')
        ax3.plot(X_sup[:min_len], S_sup[:min_len], 'g-', lw=2)
        ax3.invert_yaxis()
        ax3.set_xlabel("Horizontal Position (m)", fontsize=12)
        ax3.set_ylabel("Shear Stress $\\tau_{eq}$ (kPa)", fontsize=12, color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(X_sup[:min_len], T_dist[:min_len], 'r--', lw=2.5, label='Internal Tension $T(x)$')
        ax3_twin.set_ylabel("Track Tension T (kN)", fontsize=12, color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        ax3.set_title(f"Equivalent Shear Stress & Tension Accumulation (T1={T1:.1f} kN $\\rightarrow$ Tu={Tu:.1f} kN)", fontsize=14)
        
        lines_1, labels_1 = ax3.get_legend_handles_labels()
        lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    model = TrackedVehicleModel()
    model.solve()