import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de Rastrigin
def rastrigin_function(x):
    return 10 * len(x) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

# Parâmetros do PSO
n_particles = 30        # Número de partículas
n_iterations = 100      # Número de iterações
dim = 2                 # Dimensão do problema (2D)
w = 0.5                 # Inércia
c1 = 1.5                # Constante cognitiva
c2 = 1.5                # Constante social

# Inicialização de posições e velocidades
particles_position = np.random.uniform(-5.12, 5.12, (n_particles, dim))
particles_velocity = np.random.uniform(-1, 1, (n_particles, dim))
personal_best_position = particles_position.copy()
personal_best_value = np.array([rastrigin_function(p) for p in particles_position])

# Melhor posição global inicial
global_best_position = personal_best_position[np.argmin(personal_best_value)]
global_best_value = np.min(personal_best_value)

# Loop do PSO
for i in range(n_iterations):
    for j in range(n_particles):
        # Avaliação da função objetivo
        fitness = rastrigin_function(particles_position[j])

        # Atualiza o melhor pessoal
        if fitness < personal_best_value[j]:
            personal_best_value[j] = fitness
            personal_best_position[j] = particles_position[j]

    # Atualiza o melhor global
    if np.min(personal_best_value) < global_best_value:
        global_best_value = np.min(personal_best_value)
        global_best_position = personal_best_position[np.argmin(personal_best_value)]

    # Atualiza as velocidades e posições das partículas
    for j in range(n_particles):
        inertia = w * particles_velocity[j]
        cognitive = c1 * np.random.rand() * (personal_best_position[j] - particles_position[j])
        social = c2 * np.random.rand() * (global_best_position - particles_position[j])
        particles_velocity[j] = inertia + cognitive + social
        particles_position[j] += particles_velocity[j]

# Configuração do gráfico 3D da função e da melhor posição final
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Melhor posição final encontrada pelo PSO na Função de Rastrigin')

# Plot da melhor posição final encontrada
ax.scatter(global_best_position[0], global_best_position[1], global_best_value,
           color='red', s=100, label='Melhor Posição Final')
ax.legend()

plt.show()

print("Melhor posição encontrada:", global_best_position)
print("Melhor valor encontrado:", global_best_value)
