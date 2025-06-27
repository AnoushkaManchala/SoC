# Basic syntax practice
a = 5
b = 2.5
c = "hello"
d = True

print(type(a), type(b), type(c), type(d))

# Functions practice
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(17))  # Output: True

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
