# ATM Modularization

In this activity, you will modularize the ATM application so that the codebase matches the following image:

```text
atm
├── actions
│   ├── make_deposit.py
│   └── make_withdrawal.py
├── data
│   └── accounts.csv
├── modular_atm.py
└── utils.py
```


## Instructions

Confirm that you're working inside the [atm](Unsolved/atm/) folder. Then complete the following steps:

1. Inside the [atm](Unsolved/atm/) folder, create a new folder called `actions`. Then, in the `actions` folder, create two new Python script files: `make_deposit.py` and `make_withdrawal.py`.

2. Copy the `make_deposit(account)` function from inside [modular_atm.py](Unsolved/atm/modular_atm.py) file to the `make_deposit.py` file. Once the function has been copied over, you can delete it from `modular_atm.py` file.

3. Inside the `make_deposit.py` file, import the Python library `sys`.

    ```python
    import sys
    ```

4. Repeat steps 2 and 3 for the `make_withdrawal(account)` function:
    * Move it from `modular_atm.py` file and add it to the `make_withdrawal.py` file.
    * Inside the `make_withdrawal.py` file, import the Python library `sys`.

5. In `modular_atm.py` file, remove the following import statements.

    ```python
    import csv
    from pathlib import Path
    ```

5. Then, add the following import statements at the top of the `modular_atm.py` file under the `import sys` statement.

    ```python
    # Import the load_accounts and validate_pin functions from the utils file.
    from utils import load_accounts, validate_pin
    # Import the make_deposit function from the make_deposit file.
    from actions.make_deposit import make_deposit
    # Import the make_withdrawal function from the make_withdrawal file.
    from actions.make_withdrawal import make_withdrawal
    ```

6. Inside the `atm` folder, create a Python file called `utils.py`.

7. At the top of the `utils.py` file add the following dependencies:

    ```python
    # Import the dependencies.
    import sys
    import csv
    from pathlib import Path
    ```

8. Copy the `load_accounts()` and `validate_pin(pin)` functions from inside `modular_atm.py` file into the `utils.py` file. Once they've been copied over, they can be deleted from `modular_atm.py`.

9. Confirm that your new ATM application structure matches the figure above. If it does, run the `modular_atm.py` file to confirm that the ATM application works.


---

© 2023 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
