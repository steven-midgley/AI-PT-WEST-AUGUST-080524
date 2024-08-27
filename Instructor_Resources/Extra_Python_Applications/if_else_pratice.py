# import sys

driverless_car = True
driverless_car = int(input("Say 1 for True or 0 for False: "))

# if driverless_car not in [1,0]:
#     print("Wrong input, system will shut down now!")
#     sys.exit()


if driverless_car == 1:
    driverless_car = True
elif driverless_car == 0:
    driverless_car = False
else:
    driverless_car =False
    print("Input was not 0 or 1 so we will keep it False!")


if driverless_car == True:
    # Do something
    print("Oh no! The driver's asleep! What do we do?!")
    print()
    print("All is good. Auto-pilot initiated.")
else:
    # Do something else
    print("Oh no! The driver's asleep! MAYDAY! MAYDAY!")
