import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
import random

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


st.set_page_config(page_title="实时MCMC动画", layout="wide")
st.title("🎬 MCMC 实时采样演示")

mu = st.sidebar.slider("目标分布均值 μ", -5.0, 5.0, 0.0, 0.1)
sigma = st.sidebar.slider("标准差 σ", 0.1, 5.0, 1.0, 0.1)
n_samples = st.sidebar.slider("样本数量", 10, 200, 100, 10)
jump_sigma = st.sidebar.slider("跳跃分布标准差", 0.01, 2.0, 0.5, 0.01)
speed = st.sidebar.slider("更新间隔 (ms)", 10, 500, 50, 10)

def target_pdf(x):
    return norm(mu, sigma).pdf(x)

def metropolis(current, jump_sigma, seed=None):
    rng = np.random.default_rng(seed)
    proposal = current + rng.normal(0, jump_sigma)
    acceptance = min(1, target_pdf(proposal) / target_pdf(current))
    if rng.uniform() < acceptance:
        return proposal
    else:
        return current

if st.button("开始采样"):
    samples = []
    current = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    placeholder = st.empty()

    for i in range(n_samples):
        ran=random.randint(1, 100)
        current = metropolis(current, jump_sigma, seed=i+ran)
        samples.append(current)

        ax1.cla()
        ax1.plot(range(len(samples)), samples, lw=2, color='blue')
        ax1.set_xlim(0, n_samples)
        ax1.set_ylim(min(samples)-1, max(samples)+1)
        ax1.set_xlabel("迭代次数")
        ax1.set_ylabel("变量值")
        ax1.set_title(f"采样轨迹（迭代 {i+1}）")


        ax2.cla()
        ax2.hist(samples, bins=60, range=(mu-5*sigma, mu+5*sigma),
                 color="skyblue", alpha=0.7, density=True)
        xs = np.linspace(mu-5*sigma, mu+5*sigma, 200)
        ax2.plot(xs, norm(mu, sigma).pdf(xs),
                 'r--', label="目标分布")
        ax2.set_xlabel("x 值")
        ax2.set_ylabel("频率")
        ax2.set_xlim(mu-5*sigma, mu+5*sigma)
        ax2.set_ylim(0, 0.6/sigma)
        ax2.set_title("样本分布直方图")
        ax2.legend()

        placeholder.pyplot(fig)
        time.sleep(speed/1000)
