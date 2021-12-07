### Notes of CLF & CBF

#### Introductions

Control Lyapunov Function和Control Barrier Function常应用于safety-critical control problems。作用分别是stability和set invariance。

#### Method

对于一个时不变的control affine system，其形式为：
$$
\dot{x}=f(x)+g(x) us
$$
其中$f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$，$g: \mathbb{g}^{n} \rightarrow \mathbb{R}^{n*m}$，假设$x_e=0$是平衡点。

##### CLF

设$V(x): \mathbb{R}^{n} \rightarrow \mathbb{R}$是一个连续可微的函数，如果存在常数$c>0$，有

（1）$\Omega_{c}:=\left\{x \in \mathbb{R}^{n}: V(x) \leq c\right\}$，V(x)的一个sublevel set是有界的。

（2）$V(x)>0$ for all $s \in \mathbb{R}^{n} \backslash\left\{x_{e}\right\}, \quad V\left(x_{e}\right)=0$（正定）

（3）$\inf _{u \in U} \dot V(x, u)<0$ for all $x \in \Omega_{c} \backslash\left\{x_{e}\right\}$（负定）

那么V(x)是一个Control Lyapunov Function，每个在$\Omega_c$中的状态都会渐进稳定至$x_e$

V(x)的微分可以表示为$\dot{V}(x, u)=\nabla V(x) \cdot \dot{x}\begin{aligned}\\
&=\nabla V(x) \cdot f(x)+\nabla V(x) \cdot g(x) u \\
&=L_{f} V(x)+L_{g} V(x) u 
\end{aligned}$

由于上面的式子只表示了CLF的稳定性，没有表达其稳定速度，于是有了Exponentially Stabilizing Control Lyapunov Function。

如果存在常数$\lambda >0$，有
$$
\inf _{u \in U} \dot{V}(x, u)+\lambda V(x) \leq 0
$$
那么V(x)就是一个exponentially stabilizing CLF，$\lambda$是其decay rate的上界。

<img src="/Users/zhoujingyuan/Library/Application Support/typora-user-images/image-20211206115853359.png" alt="image-20211206115853359" style="zoom:33%;" />

##### CBF

设$B(x): D\subset\mathbb{R}^{n} \rightarrow \mathbb{R}$是一个连续可微方程，它的zero-superlevel set为C。对于所有$x \in \partial C$，$\nabla B(x) \neq 0$ ，上式中的C满足：
$$
\begin{aligned}
\mathcal{C} &=\left\{x \in D \subset \mathbb{R}^{n}: B(x) \geq 0\right\} \\
\partial \mathcal{C} &=\left\{x \in D \subset \mathbb{R}^{n}: B(x)=0\right\} \\
\operatorname{Int}(\mathcal{C}) &=\left\{x \in D \subset \mathbb{R}^{n}: B(x)>0\right\}
\end{aligned}
$$
如果存在一个扩展类$\mathcal{K}_{\infty}$函数$\alpha$，满足：
$$
\sup _{u \in U}\left[L_{f} B(x)+L_{g} B(x) u\right]+\alpha(B(x)) \geq 0
$$
那么对于所有$x\in D$，$B(x)$是一个CBF，若符合上述的约束，那么集合C就是一个安全集合(safe set)。

常常将$\alpha ()$取为常数$\gamma$，于是其就变为decay rate的下界。

<img src="/Users/zhoujingyuan/Library/Application Support/typora-user-images/image-20211206122610772.png" alt="image-20211206122610772" style="zoom:33%;" />

在另一种情况下，若
$$
\inf _{x \in \operatorname{Int}(\mathcal{C})} B(x) \geq 0, \quad \lim _{x \rightarrow \partial \mathcal{C}} B(x)=\infty
$$
则control barrier function的条件变为：
$$
\inf _{u \in U}\left[L_{f} B(x)+L_{g} B(x) u\right] \leq \alpha\left(\frac{1}{B(x)}\right)
$$
这时候的cbf可能更符合某些情况。



##### CBF-CLF-QP

可以使用二次规划(QP)来解决CBF-CLF控制问题，cost如下所示
$$
\underset{u: \text { control input }\\ \delta: \text{slack variable}}{\operatorname{argmin}}\left(u-u_{\text {ref }}\right)^{T} H\left(u-u_{\text {ref }}\right)+p \delta^{2}
$$
满足以下约束：
$$
\begin{gathered}
L_{f} V(x)+L_{g} V(x) u+\lambda V(x) \leq \delta \\
L_{f} B(x)+L_{g} B(x) u+\gamma B(x) \geq 0 \\
u \in U
\end{gathered}
$$


### 