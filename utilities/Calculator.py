def simple_calculator(num1, num2, operation):
    """
    Performs a basic arithmetic operation on two numbers.

    Args:
        num1 (float): The first number.
        num2 (float): The second number.
        operation (str): The desired operation ('+', '-', '*', '/').

    Returns:
        float or str: The result of the operation, or an error message if invalid.
    """
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        if num2 != 0:
            return num1 / num2
        else:
            return "Error: Division by zero is not allowed."
    else:
        return "Error: Invalid operation. Please use '+', '-', '*', or '/'."

# --- Main part of the program to get input and display results ---
if __name__ == "__main__":
    print("Welcome to the Simple Calculator!")

    try:
        # Get the first number from the user
        number1 = float(input("Enter the first number: "))

        # Get the operation from the user
        op = input("Enter an operation (+, -, *, /): ")

        # Get the second number from the user
        number2 = float(input("Enter the second number: "))

        # Call the calculator function with the inputs
        result = simple_calculator(number1, number2, op)

        # Display the result
        print(f"The result is: {result}")

    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")