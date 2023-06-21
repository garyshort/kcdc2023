import matplotlib.pyplot as plt

# Define the observed measure
observed_m = 2.6

# Define the range of guesses
guesses = [i*0.1 for i in range(int(5.2/0.1) + 1)]

# Calculate the sum of squared errors for each guess
errors = [(guess - observed_m)**2 for guess in guesses]

# Calculate the minimum error and its corresponding guess
min_error = min(errors)
min_index = errors.index(min_error)
best_guess = guesses[min_index]

# Plot the results
plt.plot(guesses, errors)
plt.axhline(y=min_error, color='r', linestyle='--',
            label=f'Minimum Error: {min_error}')
plt.axvline(x=observed_m, color='g', linestyle='--',
            label=f'Observed Value: {observed_m}')
plt.xlabel('Guess')
plt.ylabel('Sum of Squared Errors')
plt.title('Sum of Squared Errors for Different Guesses')
plt.legend()
plt.show()
