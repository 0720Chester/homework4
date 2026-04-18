import taichi as ti
import taichi.math as tm

# 初始化Taichi，使用GPU加速（无GPU则自动切换CPU）
ti.init(arch=ti.gpu)

# 画布分辨率
WIDTH = 800
HEIGHT = 600

# 像素缓冲区：存储渲染结果
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# Phong光照可调参数（与UI绑定）
ka = ti.field(ti.f32, shape=())  # 环境光系数
kd = ti.field(ti.f32, shape=())  # 漫反射系数
ks = ti.field(ti.f32, shape=())  # 镜面高光系数
shininess = ti.field(ti.f32, shape=())  # 高光指数

# 初始化默认参数
ka[None] = 0.2
kd[None] = 0.7
ks[None] = 0.5
shininess[None] = 32.0

# 固定场景参数
CAMERA_POS = tm.vec3(0.0, 0.0, 5.0)    # 摄像机位置
LIGHT_POS = tm.vec3(2.0, 3.0, 4.0)     # 点光源位置
LIGHT_COLOR = tm.vec3(1.0, 1.0, 1.0)   # 光源颜色（白色）
BG_COLOR = tm.vec3(0.0, 0.2, 0.3)      # 背景颜色（深青色）

# 球体参数
SPHERE_CENTER = tm.vec3(-1.2, -0.2, 0.0)
SPHERE_RADIUS = 1.2
SPHERE_COLOR = tm.vec3(0.8, 0.1, 0.1)  # 深红色

# 圆锥参数
CONE_TIP = tm.vec3(1.2, 1.2, 0.0)      # 圆锥顶点
CONE_BASE_Y = -1.4                    # 圆锥底面高度
CONE_RADIUS = 1.2                     # 底面半径
CONE_COLOR = tm.vec3(0.6, 0.2, 0.8)    # 紫色


@ti.func
def ray_sphere_intersect(ray_origin, ray_dir, center, radius):
    """
    光线-球体求交函数
    返回：(是否相交, 交点距离t)
    """
    oc = ray_origin - center
    a = tm.dot(ray_dir, ray_dir)
    b = 2.0 * tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c

    t = -1.0
    hit = False
    if discriminant >= 0:
        t = (-b - tm.sqrt(discriminant)) / (2.0 * a)
        if t > 0:
            hit = True
    return hit, t


@ti.func
def ray_cone_intersect(ray_origin, ray_dir, tip, base_y, radius):
    """
    光线-圆锥求交函数（有限圆锥）
    返回：(是否相交, 交点距离t)
    """
    t = -1.0
    hit = False
    height = tip.y - base_y
    tan_theta = radius / height

    ox, oy, oz = ray_origin
    dx, dy, dz = ray_dir
    tx, ty, tz = tip

    # 圆锥数学方程求解
    a = dx*dx + dz*dz - (dy*dy) * (tan_theta**2)
    b = 2 * ((ox - tx)*dx + (oz - tz)*dz) - 2 * (oy - ty)*dy * (tan_theta**2)
    c = (ox - tx)**2 + (oz - tz)**2 - (oy - ty)**2 * (tan_theta**2)

    discriminant = b*b - 4*a*c
    if discriminant >= 0:
        t0 = (-b - tm.sqrt(discriminant)) / (2*a)
        if t0 > 0:
            y_intersect = oy + t0*dy
            # 限制圆锥高度范围
            if base_y <= y_intersect <= tip.y:
                t = t0
                hit = True
    return hit, t


@ti.func
def get_sphere_normal(point, center):
    """计算球体表面法向量"""
    return tm.normalize(point - center)


@ti.func
def get_cone_normal(point, tip, base_y, radius):
    """计算圆锥表面法向量"""
    height = tip.y - base_y
    tan_theta = radius / height
    x, y, z = point
    tx, ty, tz = tip

    nx = x - tx
    ny = (y - ty) * (tan_theta**2)
    nz = z - tz
    return tm.normalize(tm.vec3(nx, -ny, nz))


@ti.func
def phong_shading(point, normal, obj_color):
    """
    Phong光照模型计算
    输入：交点坐标、法向量、物体基础颜色
    输出：最终像素颜色
    """
    # 1. 计算方向向量
    light_dir = tm.normalize(LIGHT_POS - point)   # 光线方向 L
    view_dir = tm.normalize(CAMERA_POS - point)   # 视线方向 V
    reflect_dir = tm.normalize(2 * tm.dot(normal, light_dir) * normal - light_dir)  # 反射方向 R

    # 2. 环境光
    ambient = ka[None] * LIGHT_COLOR * obj_color

    # 3. 漫反射（Lambert定律）
    diff_dot = tm.max(0.0, tm.dot(normal, light_dir))
    diffuse = kd[None] * diff_dot * LIGHT_COLOR * obj_color

    # 4. 镜面高光
    spec_dot = tm.max(0.0, tm.dot(reflect_dir, view_dir))
    specular = ks[None] * (spec_dot ** shininess[None]) * LIGHT_COLOR

    # 叠加三个分量
    final_color = ambient + diffuse + specular
    return final_color


@ti.kernel
def render():
    """主渲染内核：光线投射+深度测试+着色"""
    for i, j in pixels:
        # 屏幕坐标归一化（-1~1）
        u = (i / WIDTH) * 2.0 - 1.0
        v = (j / HEIGHT) * 2.0 - 1.0
        # 修正宽高比
        u *= WIDTH / HEIGHT

        # 构建光线：起点=摄像机，方向=指向像素点
        ray_dir = tm.normalize(tm.vec3(u, v, -1.0))

        # 光线与两个物体求交
        hit_sphere, t_sphere = ray_sphere_intersect(CAMERA_POS, ray_dir, SPHERE_CENTER, SPHERE_RADIUS)
        hit_cone, t_cone = ray_cone_intersect(CAMERA_POS, ray_dir, CONE_TIP, CONE_BASE_Y, CONE_RADIUS)

        # 深度测试：选择最近的有效交点
        min_t = -1.0
        hit_obj = 0  # 0=无碰撞，1=球体，2=圆锥
        if hit_sphere and t_sphere > 0:
            min_t = t_sphere
            hit_obj = 1
        if hit_cone and t_cone > 0:
            if min_t < 0 or t_cone < min_t:
                min_t = t_cone
                hit_obj = 2

        # 着色
        if hit_obj != 0 and min_t > 0:
            # 计算交点坐标
            hit_point = CAMERA_POS + min_t * ray_dir
            normal = tm.vec3(0.0)
            obj_color = tm.vec3(0.0)

            # 获取对应物体的法向量和颜色
            if hit_obj == 1:
                normal = get_sphere_normal(hit_point, SPHERE_CENTER)
                obj_color = SPHERE_COLOR
            else:
                normal = get_cone_normal(hit_point, CONE_TIP, CONE_BASE_Y, CONE_RADIUS)
                obj_color = CONE_COLOR

            # Phong着色
            pixels[i, j] = phong_shading(hit_point, normal, obj_color)
        else:
            # 无碰撞：背景色
            pixels[i, j] = BG_COLOR


def main():
    # 创建窗口
    window = ti.ui.Window("Phong光照模型实验", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        # 渲染画面
        render()
        canvas.set_image(pixels)

        # 绘制UI控制面板
        with gui.sub_window("参数调节", 0.05, 0.05, 0.3, 0.4):
            ka[None] = gui.slider_float("Ka(环境光系数)", ka[None], 0.0, 1.0)
            kd[None] = gui.slider_float("Kd(漫反射系数)", kd[None], 0.0, 1.0)
            ks[None] = gui.slider_float("Ks(高光系数)", ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess(高光指数)", shininess[None], 1.0, 128.0)

        window.show()


if __name__ == "__main__":
    main()
