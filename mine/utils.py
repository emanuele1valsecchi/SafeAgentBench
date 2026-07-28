def print_separator():
    print("\n" + "=" * 100 + "\n")

def yn_question(question):
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

def req_not_empty_value(question, error_message="The value inserted is not allowed, please provide a valid input"):
    d = ""

    while not d:
        d = input( question ).strip()

        if d == "":
            print(f"{error_message} \n")
            d = ""

    return d