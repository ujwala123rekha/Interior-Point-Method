import numpy as np
import matplotlib.pyplot as plt

# Objective Function
def objective(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

# Constraint: x1 + x2 <= 1
def constraint(x):
    return 1 - (x[0] + x[1])

# Optimizer 1: Gradient Descent
def gradient_descent(x0, lr=0.1, iters=50):

    x = x0.copy()
    history = []

    for i in range(iters):

        grad = gradient(x)
        x = x - lr * grad

        history.append(x.copy())

    return x, np.array(history)

# Optimizer 2: Projected Gradient
def projected_gradient(x0, lr=0.1, iters=50):

    x = x0.copy()
    history = []

    for i in range(iters):

        grad = gradient(x)
        x = x - lr * grad

        # projection if constraint violated
        if x[0] + x[1] > 1:
            x = x / (x[0] + x[1])

        history.append(x.copy())

    return x, np.array(history)

# Optimizer 3: Interior Point Method
def interior_point(x0, mu=0.1, lr=0.05, iters=50):

    x = x0.copy()
    history = []

    for i in range(iters):

        # ensure constraint value doesn't become zero
        c = max(constraint(x), 1e-6)

        grad_f = gradient(x)

        # barrier gradient
        barrier_grad = mu * np.array([1/c, 1/c])

        grad = grad_f + barrier_grad

        x = x - lr * grad

        history.append(x.copy())

    return x, np.array(history)

# Run Experiments
x0 = np.array([0.5, 0.3])

gd_sol, gd_hist = gradient_descent(x0)
pgd_sol, pgd_hist = projected_gradient(x0)
ip_sol, ip_hist = interior_point(x0)

print("Gradient Descent Solution:", gd_sol)
print("Projected Gradient Solution:", pgd_sol)
print("Interior Point Solution:", ip_sol)

# Compute Loss Values
def compute_loss(history):

    loss = []

    for x in history:
        loss.append(objective(x))

    return loss

loss_gd = compute_loss(gd_hist)
loss_pgd = compute_loss(pgd_hist)
loss_ip = compute_loss(ip_hist)

# Plot Convergence Graph
plt.figure(figsize=(8,5))

plt.plot(loss_gd, label="Gradient Descent")
plt.plot(loss_pgd, label="Projected Gradient")
plt.plot(loss_ip, label="Interior Point")

plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Optimizer Convergence Comparison")

plt.legend()
plt.grid()

plt.show()

# Plot Optimization Paths
plt.figure(figsize=(6,6))

gd_hist = np.array(gd_hist)
pgd_hist = np.array(pgd_hist)
ip_hist = np.array(ip_hist)

plt.plot(gd_hist[:,0], gd_hist[:,1], 'r-o', label="Gradient Descent")
plt.plot(pgd_hist[:,0], pgd_hist[:,1], 'g-s', label="Projected Gradient")
plt.plot(ip_hist[:,0], ip_hist[:,1], 'b-^', label="Interior Point")

# constraint boundary
x = np.linspace(0,1,100)
y = 1 - x

plt.plot(x, y, 'k--', label="Constraint Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Optimization Paths")

plt.legend()
plt.grid()

plt.show()