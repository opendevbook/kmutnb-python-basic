# Python 3.12 Object-Oriented Programming (OOP) Tutorial

## Introduction to OOP in Python 3.12
Object-Oriented Programming (OOP) is a programming paradigm based on the concept of objects. Python 3.12 continues to improve its OOP capabilities, providing a robust environment for designing modular, reusable, and maintainable code.

### Key OOP Concepts:
- **Class**: A blueprint for creating objects.
- **Object**: An instance of a class.
- **Encapsulation**: Hiding internal details of an object and only exposing necessary functionalities.
- **Inheritance**: Creating new classes from existing ones.
- **Polymorphism**: Using a unified interface for different data types.
- **Abstraction**: Hiding complex implementation details and exposing only the essential features.

## Defining a Class and Creating Objects
```python
class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def display_info(self):
        return f"{self.year} {self.brand} {self.model}"

# Creating an object
my_car = Car("Toyota", "Corolla", 2023)
print(my_car.display_info())
```

## Encapsulation in Python
Encapsulation restricts direct access to some of an object’s components. This is done using private variables and methods.
```python
class BankAccount:
    def __init__(self, account_number, balance):
        self.account_number = account_number
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance

# Usage
account = BankAccount("123456", 1000)
account.deposit(500)
print(account.get_balance())  # 1500
```

## Inheritance
Inheritance allows a class to derive properties and methods from another class.
```python
class ElectricCar(Car):
    def __init__(self, brand, model, year, battery_capacity):
        super().__init__(brand, model, year)
        self.battery_capacity = battery_capacity

    def display_info(self):
        return f"{super().display_info()} with {self.battery_capacity} kWh battery"

# Usage
my_tesla = ElectricCar("Tesla", "Model 3", 2023, 75)
print(my_tesla.display_info())
```

## Polymorphism
Polymorphism allows different classes to be treated through a common interface.
```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

# Polymorphic function
def make_sound(animal):
    print(animal.speak())

# Usage
make_sound(Dog())  # Woof!
make_sound(Cat())  # Meow!
```

## Abstraction
Abstraction is implemented using abstract base classes (ABCs) in Python.
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Usage
my_dog = Dog()
print(my_dog.speak())  # Woof!
```

## New Features in Python 3.12 Relevant to OOP
Python 3.12 introduces various enhancements, including:
- **Performance improvements** for method calls.
- **More efficient error messages** for better debugging.
- **Refinements to `typing` module** for type hinting, which is helpful in OOP.

Example with type hints:
```python
from typing import Union

def multiply(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x * y

print(multiply(5, 3.2))
```

## Conclusion
Object-Oriented Programming in Python 3.12 allows for structured and maintainable code. With the core principles of classes, encapsulation, inheritance, polymorphism, and abstraction, developers can build scalable applications efficiently.

