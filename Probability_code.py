import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Coin Toss and Dice Simulation
def simulate_coin_tosses(trials=10000):
    heads = 0
    tails = 0
    for _ in range(trials):
        toss = random.choice(['Heads', 'Tails'])
        if toss == 'Heads':
            heads += 1
        else:
            tails += 1
    print("Coin Toss Simulation:")
    print(f"Probability of Heads: {heads/trials:.4f}")
    print(f"Probability of Tails: {tails/trials:.4f}")
    print('-'*40)

def simulate_dice_sum_7(trials=10000):
    count_sum_7 = 0
    for _ in range(trials):
        die1 = random.randint(1,6)
        die2 = random.randint(1,6)
        if die1 + die2 == 7:
            count_sum_7 += 1
    print("Dice Roll Simulation:")
    print(f"Probability of sum 7: {count_sum_7/trials:.4f}")
    print('-'*40)

# 2. Probability of at least one 6 in 10 rolls
def probability_at_least_one_six(trials=10000):
    success = 0
    for _ in range(trials):
        rolls = [random.randint(1,6) for _ in range(10)]
        if 6 in rolls:
            success += 1
    print("At Least One 6 in 10 Rolls:")
    print(f"Estimated Probability: {success/trials:.4f}")
    print('-'*40)

# 3. Conditional Probability and Bayes' Theorem
def simulate_balls(trials=1000):
    colors = ['Red'] * 5 + ['Green'] * 7 + ['Blue'] * 8
    previous_color = None
    count_blue_followed_by_red = 0
    count_blue = 0

    for _ in range(trials):
        current_color = random.choice(colors)
        if previous_color == 'Blue':
            count_blue += 1
            if current_color == 'Red':
                count_blue_followed_by_red += 1
        previous_color = current_color

    if count_blue > 0:
        conditional_probability = count_blue_followed_by_red / count_blue
    else:
        conditional_probability = 0

    print("Conditional Probability Simulation:")
    print(f"Probability of Red given previous Blue: {conditional_probability:.4f}")
    print('-'*40)
    # Bayes verification would need theoretical values, here we are estimating only.

# 4. Discrete Random Variable Simulation
def discrete_random_variable():
    values = [1, 2, 3]
    probabilities = [0.25, 0.35, 0.4]
    sample = np.random.choice(values, size=1000, p=probabilities)

    mean = np.mean(sample)
    variance = np.var(sample)
    std_dev = np.std(sample)

    print("Discrete Random Variable Sample:")
    print(f"Empirical Mean: {mean:.4f}")
    print(f"Empirical Variance: {variance:.4f}")
    print(f"Empirical Standard Deviation: {std_dev:.4f}")
    print('-'*40)

# 5. Continuous Random Variable (Exponential Distribution)
def exponential_distribution_simulation():
    data = np.random.exponential(scale=5, size=2000)

    plt.figure(figsize=(10,6))
    sns.histplot(data, bins=30, kde=True, stat="density", color='skyblue', edgecolor='black')
    plt.title('Exponential Distribution Histogram with PDF Overlay')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# 6. Central Limit Theorem Simulation
def central_limit_theorem_simulation():
    population = np.random.uniform(0, 1, 10000)
    
    # Plotting population
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(population, bins=30, color='orange', kde=True)
    plt.title('Uniform Population Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Sampling
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population, size=30)
        sample_means.append(np.mean(sample))
    
    # Plotting sample means
    plt.subplot(1,2,2)
    sns.histplot(sample_means, bins=30, color='green', kde=True)
    plt.title('Sample Means Distribution (n=30)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Running All Simulations
if __name__ == "__main__":
    simulate_coin_tosses()
    simulate_dice_sum_7()
    probability_at_least_one_six()
    simulate_balls()
    discrete_random_variable()
    exponential_distribution_simulation()
    central_limit_theorem_simulation()
