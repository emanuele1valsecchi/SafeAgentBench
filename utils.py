# === LOG HANDLER === #
import logging

logging.basicConfig(
    filename='simulation_results.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def save_in_log(text : str = None):
    logging.info(text)

def print_log(text : str = None, save_to_log : bool = True):
    if not text:
        return

    print(text)

    if save_to_log:
        save_in_log(text)

def input_log(text: str = None, save_to_log: bool = True) -> str:
    """Acts like standard input() but logs the prompt + user response."""
    if not text:
        return

    user_response = input(text)
    
    if save_to_log:
        logging.info(f"{text}{user_response}")
        
    return user_response

# === INPUT/OUTPUT === #

def format_column_content(*, 
        content : list[str], 
        column : int = 3,
        numbers = False
    ):
    content_len = len(content)

    num_rows = (content_len + column - 1) // column
    
    content_formatted = ""
    
    for i in range(num_rows):
        row_items = []
        
        for col_index in range(i, content_len, num_rows):
            if numbers:
                row_items.append(f"({col_index + 1}) {content[col_index]:<18}")
            else:
                row_items.append(f"- {content[col_index]:<18}")
        
        if row_items:
            content_formatted += "  " + "  ".join(row_items) + "\n"

    return content_formatted

def print_separator(character : str = "=", title : str = None, char_numb = 60, save_to_log : bool = True):
    if not title:
        print_log(
            text = "\n" + character * char_numb + "\n", 
            save_to_log = save_to_log)
    else:
        char_numb = int((char_numb - len(title) - 4) / 2)
        print_log(
            text = "\n" + character * char_numb + f" [{title}] " + character * char_numb + "\n", 
            save_to_log = save_to_log)

def yn_question(question: str, save_to_log : bool = True) -> bool:
    resp = ""

    while not resp:
        resp = input_log(text = f"(Y/n) {question} : ", save_to_log = save_to_log).strip()

        if resp.lower() == "y" or resp.lower() == "yes":
            return True
        elif resp.lower() == "n" or resp.lower() == "no":
            return False
        else:
            print_log(text = f"\nAnswer can only be 'y', 'yes', 'n' or 'no', cannot accept '{resp}'\n", save_to_log = save_to_log)
            resp = ""
            continue

    return False

def req_not_empty_value(
        question: str, 
        error_message: str="The value inserted is not allowed, please provide a valid input",
        save_to_log : bool = True
    ):
    d = ""

    while not d:
        d = input_log(text = question, save_to_log = save_to_log).strip()

        if d == "":
            print_log(text = f"{error_message} \n", save_to_log = save_to_log)
            d = ""

    return d

def wait_ui(
        text : str = "", 
        end_message: str = "Press [Enter] to continue",
        save_to_log : bool = True):
    """Wait for user input: wait for enter key pressed by the user"""
    if end_message:
        input_log(text=f"{text}\n{end_message}", save_to_log=save_to_log)
    else:
        input_log(text=f"{text}", save_to_log=save_to_log)

def quit_program(text : str = "", end_message: str = "Press [Enter] to close the program"):
    wait_ui(text, end_message)
    quit()