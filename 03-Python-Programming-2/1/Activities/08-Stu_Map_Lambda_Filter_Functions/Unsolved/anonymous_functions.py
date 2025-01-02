""" Anonymous Functions"""

# 1. Get the even numbers from a list using the filter and lambda functions.
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# 2. Use the map and lambda functions to add the numbers from both lists.
numbers1 = [1, 2, 3, 4, 5]
numbers2 = [10, 20, 30, 40, 50]
merged = list(map(lambda x, y: x + y, numbers1, numbers2))
print(merged)


# 3. Use the map and lambda functions to split the following sentence into words.
sentence = "My favorite hobby is coding in Python"
