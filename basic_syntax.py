# Basic syntax practice
a = 5
b = 2.5
c = "hello"
d = True

print(type(a), type(b), type(c), type(d))



# Eroor handling
try:
    x = int(input("Enter a number: "))
    print(100 / x)
except ZeroDivisionError:
    print("Can't divide by zero!")
except ValueError:
    print("Invalid input. Please enter a number.")
finally:
    print("Execution complete.")
