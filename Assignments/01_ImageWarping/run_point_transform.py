import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # type: ignore # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy() # type: ignore
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts : np.ndarray, target_pts : np.ndarray, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """

    #使用RBF算法进行图像变换
    #G@A=F

    #RBF径向基函数
    coeff = 0.001
    g = lambda ref, x: np.exp(-coeff *  np.power(np.linalg.norm(ref-x), 1))
    
    #计算G
    pts_count = len(source_pts)
    G = np.ndarray((pts_count+2, pts_count+2))
    for i in range(pts_count):
        for j in range(pts_count):
            G[i,j] = g(source_pts[i], source_pts[j])
    
    G[0:pts_count, pts_count:pts_count+2] = source_pts
    G[pts_count:pts_count+2, 0:pts_count] = source_pts.transpose()
    G[pts_count:pts_count+2, pts_count:pts_count+2] = 0

    #计算F
    F = np.vstack([target_pts, np.zeros((2, 2))])

    #计算A
    A = np.linalg.solve(G, F)

    #构造warped image
    shape = image.shape
    warped_image = np.full(shape, (255,255,255))
    warped_shape =warped_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            src = np.array([j, i])
            w = np.zeros(pts_count + 2)
            for k in range(pts_count):
                w[k] = g(source_pts[k], src)
            w[pts_count:pts_count+2] = src
            dst = np.round(w@A).astype(int)
            if\
                dst[1] >= 0 and\
                dst[0] >= 0 and\
                dst[1] < warped_shape[0] and\
                dst[0] < warped_shape[1]:
                warped_image[dst[1],dst[0]] \
                     = image[src[1],src[0]]

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
