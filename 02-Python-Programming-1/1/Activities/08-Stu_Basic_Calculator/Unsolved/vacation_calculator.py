# Ask the user what their vacation budget is and convert it to a float
budget = float(input("What is your total vacation budget? ") or 51351424823)
print(f"\nBudget: ${budget}\n")

# Ask the user how much of their budget should be spent on flights and
# accommodation and convert it to a float
flights_accommodation = float(
    input("What is your budget for the flights and accommodation? ") or 5514323
)
print(f"\nFlight accommodations: ${flights_accommodation}\n")

# Ask the user how much they would like to spend per day and convert it to a float
daily_budget = float(input("What is your preferred daily budget? ") or 5514323)
print(f"\nDaily Budget: ${daily_budget}\n")

# Ask the user how many days they will go on vacation and convert it to an
# integer
vacation_days = float(input("How many days will you go on vacation? ") or 365)
print(f"\nVacation Days: {vacation_days}\n")

# Ask the user the currency exchange rate for the country they're visiting and
# convert it to a float
exchange_rate = float(input("What is the currency exchange rate? ") or 0.67)
print(f"\nExchange Rate: {exchange_rate}\n")

# Ask the user for the radius distance they're willing to walk from their
# hotel and convert it to a float
distance = float(
    input(
        "What is the radius distance you're willing to walk from your hotel (in meters)? "
    )
    or 666
)
print(f"\ndistance willing to walk: {distance} meters\n")

# Calculate the budget remaining after flights and accommodation
remaining_funds = float(budget - flights_accommodation)
print(f"\nFunds after flight & accommodations: ${remaining_funds}.\n")

# Calculate the remaining budget in local currency amount
local_budget = float(remaining_funds * exchange_rate)
print(f"\nLocal currency budget: ${local_budget}\n")

# Calculate the budget per day in the local currency
daily_local_budget = float(local_budget // 7)
print(f"\nDaily budget in local currency: ${daily_local_budget}\n")

# Calculate the budget per day in the local currency, ignoring cents
rounded_local_budget = int(local_budget // 7)
print(f"\nDaily budget in local currency (rounded): ${rounded_local_budget}\n")

# Calculate the total area around the hotel the user can walk within
# Area of a circle = pi * radius ** 2
pi = 3.14159265358979323846
circle = pi * distance**2
print(f"You will only be allowed to walk within {circle}")
# Calculate the amount left from the budget if the user sticks with their
# daily budget. This is a modulus problem.
