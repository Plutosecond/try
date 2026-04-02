import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io

st.set_page_config(page_title="HMC动态采样演示", layout="wide")

st.title("🎬 HMC（Hamiltonian Monte Carlo）动态采样演示")

# ========== 左侧控制面板 ==========
st.sidebar.header("参数设置")

# 分布参数
mu = st.sidebar.slider("目标分布均值 μ", -5.0, 5.0, 0.0, 0.1)
sigma = st.sidebar.slider("标准差 σ", 0.1, 5.0, 1.0, 0.1)

# HMC 参数
eps = st.sidebar.slider("步长 ε", 0.01, 0.5, 0.1, 0.01)
L = st.sidebar.slider("轨迹步数 L", 1, 50, 20, 1)
n_samples = st.sidebar.slider("样本数量", 10, 300, 100, 10)

# 动画速度控制
speed = st.sidebar.slider("动画播放速度(ms)", 10, 300, 100, 10)

st.sidebar.markdown("---")
st.sidebar.write("点击按钮生成动画 👇")

# ========== 目标函数 ==========
def U(q):
    return 0.5 * ((q - mu) / sigma)**2

def grad_U(q):
    return (q - mu) / (sigma**2)

# ========== HMC 采样器 ==========
def HMC(U, grad_U, eps, L, current_q, seed):
    rng = np.random.default_rng(seed)
    q = current_q
    p = rng.normal(0, 1)
    current_p = p

    p -= eps * grad_U(q) / 2
    for _ in range(L):
        q += eps * p
        if _ != L - 1:
            p -= eps * grad_U(q)
    p -= eps * grad_U(q) / 2
    p = -p

    current_U = U(current_q)
    current_K = current_p**2 / 2
    proposed_U = U(q)
    proposed_K = p**2 / 2

    if rng.uniform() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q
    else:
        return current_q

# ========== 运行采样 ==========
if st.sidebar.button("开始采样并生成动画"):
    st.write("正在采样中... 请稍候 ⏳")

    samples = []
    q = 0.0
    for i in range(n_samples):
        q = HMC(U, grad_U, eps, L, current_q=q, seed=i)
        samples.append(q)
    samples = np.array(samples)

    # ========== 生成动画 ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"HMC动态采样 (μ={mu}, σ={sigma}, ε={eps}, L={L})")

    line, = ax1.plot([], [], lw=2)
    ax1.set_xlim(0, n_samples)
    ax1.set_ylim(min(samples) - 1, max(samples) + 1)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("x value")

    def update(frame):
        line.set_data(range(frame + 1), samples[:frame + 1])
        ax1.set_title(f"Iteration: {frame + 1}")

        ax2.cla()
        ax2.hist(samples[:frame + 1], bins=20, range=(mu - 5*sigma, mu + 5*sigma),
                 color="skyblue", alpha=0.7, density=True)
        xs = np.linspace(mu - 5*sigma, mu + 5*sigma, 200)
        ax2.plot(xs, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs - mu)/sigma)**2),
                 'r--', label="目标分布")
        ax2.legend()
        ax2.set_xlim(mu - 5*sigma, mu + 5*sigma)
        ax2.set_ylim(0, 0.6/sigma)
        ax2.set_title("样本分布")
        return line,

    ani = FuncAnimation(fig, update, frames=n_samples, interval=speed, blit=False)

    # 将动画保存为GIF以便前端展示
    buf = io.BytesIO()
    ani.save(buf,  writer='pillow')
    st.image(buf.getvalue(), caption="HMC采样动态演示", use_column_width=True)

    st.success("✅ 动画生成完成！")

else:
    st.info("请在左侧调整参数后点击“开始采样并生成动画”。")

