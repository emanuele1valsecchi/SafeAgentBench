def print_separator(char_numb = 60):
    print("\n" + "=" * char_numb + "\n")

def yn_question(question: str) -> bool:
    resp = ""

    while not resp:
        resp = input( f"(Y/n) {question} : " ).strip()

        if resp.lower() == "y" or resp.lower() == "yes":
            return True
        elif resp.lower() == "n" or resp.lower() == "no":
            return False
        else:
            print(f"\nAnswer can only be 'y', 'yes', 'n' or 'no', cannot accept '{resp}'\n")
            resp = ""
            continue

    return False

def req_not_empty_value(question: str, error_message: str="The value inserted is not allowed, please provide a valid input"):
    d = ""

    while not d:
        d = input( question ).strip()

        if d == "":
            print(f"{error_message} \n")
            d = ""

    return d

def wait_ui(text : str = "", end_message: str = "Press enter to continue"):
    """Wait for user input: wait for enter key pressed by the user"""
    if end_message:
        input (f"{text}\n\n{end_message}")
    else:
        input(f"{text}")

def quit_program(text : str = "", end_message: str = "Press enter to close the program"):
    wait_ui(text, end_message)
    quit()