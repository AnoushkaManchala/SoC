# Try-except-finally example
try:
    x = int(input("Enter a number: "))
    print(100 / x)
except ZeroDivisionError:
    print("Can't divide by zero!")
except ValueError:
    print("Invalid input. Please enter a number.")
finally:
    print("Execution complete.")
