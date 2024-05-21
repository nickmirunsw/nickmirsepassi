# server.py

# Import necessary libraries
import os
import sys
import time
import datetime
import threading
import logging
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread, Lock


# Global Constants
MAX_CONNECTIONS = 30
AUTH_ATTEMPTS = 5
CONSECUTIVE_LOGIN_ATTEMPTS = 3
BLOCK_DURATION = 10  # Default block duration in seconds
VALID_BLOCK_DURATIONS = range(1, 6)
MIN_CONSECUTIVE_ATTEMPTS = 1
MAX_CONSECUTIVE_ATTEMPTS = 5


# Global Variables
active_users = []  # List to store currently active users
group_chats = {}  # Dictionary to store group chat information
client_sockets = []  # List to store all connected client sockets
active_users_lock = threading.Lock()  # Lock to protect access to active_users
blocked_users = {}  # Dictionary to store blocked users and their block durations


# Initialize the logging module
logging.basicConfig(filename='server.log', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d %b %Y %H:%M:%S')
# Ensure the log file is created
open('server.log', 'a').close()


def load_credentials():
    """
    Load credentials from the 'credentials.txt' file and return a dictionary.
    The file should contain lines in the format: 'username password'.
    """
    credentials = {}
    with open("credentials.txt", "r") as file:
        for line in file:
            username, password = line.strip().split()
            credentials[username] = password
    return credentials


def block_user(client_socket, block_duration):
    # Print a message indicating that the user is blocked for a specific duration
    print(f"Too many consecutive failed login attempts. User blocked for {block_duration} seconds.")
    # Pause execution for the specified block_duration
    time.sleep(block_duration)


def log_user_info(username, client_address, client_port):
    # Get the current timestamp in the desired format
    timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')

    # Check if userlog.txt exists, and if not, create it and write the header
    if not os.path.exists("userlog.txt"):
        with open("userlog.txt", "w") as log_file:
            log_file.write("User Entry Number; Timestamp; Username; Client Address; Client Port\n")

    # Load the current user entry number
    with open("userlog.txt", "r") as log_file:
        user_entries = log_file.read().splitlines()
        if user_entries:
            # Extract the last entry and determine the last user entry number
            last_entry = user_entries[-1]
            if not last_entry.startswith("User Entry Number"):
                last_user_entry = int(last_entry.split(";")[0])
            else:
                last_user_entry = 0
        else:
            last_user_entry = 0

    # Increment the user entry number
    user_entry_number = last_user_entry + 1

    # Append the user information to the userlog.txt file
    with open("userlog.txt", "a") as log_file:
        log_file.write(f"{user_entry_number}; {timestamp}; {username}; {client_address}; {client_port}\n")


def authenticate_user(client_socket, consecutive_attempts):
    # Load user credentials from a secure source (not shown in this snippet)
    credentials = load_credentials()
    failed_attempts = 0

    # Continue attempting authentication until consecutive attempts are reached
    while failed_attempts < consecutive_attempts:
        # Receive username and password from the client
        username = client_socket.recv(1024).decode()
        password = client_socket.recv(1024).decode()

        # Check if the user is blocked
        if username in blocked_users:
            block_start_time, block_duration = blocked_users[username]
            current_time = time.time()

            # Check if the block duration has expired
            if current_time - block_start_time < block_duration:
                # Inform the client that the account is blocked
                client_socket.send(f"Your account is blocked due to multiple login failures. Please try again later.".encode())
                return None  # Return None to indicate that the user is blocked

        # Check if the provided username and password match the stored credentials
        if username in credentials and credentials[username] == password:
            # Authentication successful
            client_socket.send("Authentication successful".encode())
            print("Authentication successful. Welcome, {}!".format(username))
            return username
        else:
            failed_attempts += 1
            remaining_attempts = consecutive_attempts - failed_attempts

            # Check if there are remaining attempts
            if remaining_attempts > 0:
                # Inform the client about the authentication failure and remaining attempts
                client_socket.send(f"Authentication failed. Please retry. {remaining_attempts} attempts remaining.".encode())
                print(f"Authentication failed. Please retry. {remaining_attempts} attempts remaining.")
            else:
                # Block the user after consecutive failed attempts
                block_start_time = time.time()
                blocked_users[username] = (block_start_time, BLOCK_DURATION)

                if failed_attempts == consecutive_attempts:
                    # Inform the client that the account is blocked due to invalid password
                    client_socket.send(f"Invalid Password. Your account is blocked. Please try again later.".encode())
                else:
                    # Inform the client about the invalid password and remaining attempts
                    if failed_attempts == 0:
                        client_socket.send("Invalid Password. Please try again.".encode())
                    else:
                        client_socket.send(f"Invalid Password. Please try again. {consecutive_attempts - failed_attempts} attempts remaining.".encode())
                        time.sleep(2)  # Add a delay before allowing the user to retry
                        return None  # Add this line to exit the function immediately

    return None


def cleanup_blocked_users():
    # Remove users from the blocked list whose block duration has expired
    current_time = time.time()
    users_to_remove = [username for username, (block_start_time, block_duration) in blocked_users.items() if current_time - block_start_time >= block_duration]
    for username in users_to_remove:
        del blocked_users[username]


def create_group_chat(group_name, creator, members):
    # Create a log file for the group chat if it doesn't exist
    group_log_file = f"{group_name}_messagelog.txt"
    if not os.path.exists(group_log_file):
        with open(group_log_file, "w"):
            pass

    # Store group chat members
    group_members = [creator] + members

    return group_log_file, group_members


def handle_group_creation(client_socket, username, command):
    # Parse the command to extract group name and members
    parts = command.split(' ')
    group_name = parts[1]
    members = parts[2:]

    # Check if the group already exists
    if group_name in group_chats:
        client_socket.send(f"A group chat (Name: {group_name}) already exists.".encode())
    else:
        # Create a new group chat and add it to the dictionary
        group_log_file, group_members = create_group_chat(group_name, username, [m for m in members if m != username])
        group_chats[group_name] = {"log_file": group_log_file, "members": group_members} 
        client_socket.send(f"Group chat room has been created, room name: {group_name}, users in this room: {', '.join(group_members)}".encode())


def handle_group_message(client_socket, username, command):
    # Parse the command to extract group name and message content
    parts = command.split(' ')
    group_name = parts[1]
    message_content = ' '.join(parts[2:])

    # Check if the group exists
    if group_name not in group_chats:
        client_socket.send("The group chat does not exist.".encode())
        return

    # Retrieve information about the group
    group_info = group_chats[group_name]
    group_members = group_info["members"]

    # Check if the sender is a member of the group
    if username not in group_members:
        client_socket.send("You are not in this group chat.".encode())
        return

    # Get the current timestamp and message number
    timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')
    message_number = get_message_number(group_name)
    message_to_log = f"{message_number}; {timestamp}; {username}; {message_content}\n"

    # Append the message to the group's log file
    with open(group_info["log_file"], "a") as log_file:
        log_file.write(message_to_log)

    # Send confirmation message to the sender
    confirmation_message = f"Group chat message {message_number} at {timestamp}: {message_content}"
    client_socket.send(confirmation_message.encode())


# Function to get the message number for a specific group
def get_message_number(group_name):
    # Create a file name based on the group name
    message_number_file = f"{group_name}_message_number.txt"
    
    try:
        # Try to read the last message number from the file
        with open(message_number_file, "r") as file:
            last_message_number = int(file.read())
    except FileNotFoundError:
        # If the file is not found, set the last message number to 0
        last_message_number = 0

    # Increment the message number for the new message
    new_message_number = last_message_number + 1

    # Update the message number in the file
    with open(message_number_file, "w") as file:
        file.write(str(new_message_number))

    # Return the new message number
    return new_message_number


# Function to handle a private message
def handle_private_message(client_socket, username, command):
    # Split the command into parts
    parts = command.split(' ')
    
    # Extract recipient and message content from the command
    recipient = parts[1]
    message_content = ' '.join(parts[2:])

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')
    
    # Log the private message and get the message number
    message_number = log_private_message(username, recipient, message_content, timestamp)

    # Create a confirmation message
    confirmation_message = f"Private message {message_number} at {timestamp}: {message_content}"
    
    # Send the confirmation message to the client socket
    client_socket.send(confirmation_message.encode())


# Function to log a private message
def log_private_message(sender, recipient, message, timestamp):
    # Get the message number for the private message
    message_number = get_message_number("private_messages")
    
    # Log the message to a file (e.g., messagelog.txt)
    with open("messagelog.txt", "a") as log_file:
        log_file.write(f"{message_number}; {timestamp}; {sender} to {recipient}: {message}\n")

    # Return the message number
    return message_number


# Function to get the private message number from a file or initialize to 0 if the file is not found
def get_private_message_number():
    message_number_file = "private_message_number.txt"
    try:
        with open(message_number_file, "r") as file:
            last_message_number = int(file.read())
    except FileNotFoundError:
        last_message_number = 0

    # Increment the message number for the new message
    new_message_number = last_message_number + 1

    # Update the message number in the file
    with open(message_number_file, "w") as file:
        file.write(str(new_message_number))

    return new_message_number


# Function to handle a client's request for active users and send the information
def handle_active_user_request(client_socket, sender_username):
    # Retrieve information about active users excluding the sender
    active_users_info = get_active_users_info(sender_username)

    # If there are active users, format the data and send it to the client
    if active_users_info:
        active_user_data = "\n".join(active_users_info)
        client_socket.send(active_user_data.encode())
    else:
        # If no active users, send a message indicating that
        client_socket.send("no other active user".encode())


# Function to retrieve information about active users, excluding the sender
def get_active_users_info(sender_username):
    active_users_info = []

    # Iterate through active_users list and exclude the sender's information
    for user in active_users:
        username, client_address, client_port = user
        if username != sender_username:
            # Get the timestamp and format user information
            timestamp = datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')
            active_users_info.append(f"{username} (Active since {timestamp})")

    return active_users_info


# Function to handle logout, remove the user from active users, and send a confirmation message
def handle_logout(client_socket, username):
    global active_users

    # Find the user in the list of active users
    user_index = None
    for i, user in enumerate(active_users):
        if user[0] == username:
            user_index = i
            break

    if user_index is not None:
        # Remove the user from the list of active users
        removed_user = active_users.pop(user_index)

        # Log the user information as "N/A" for address and port
        log_user_info(username, "N/A", "N/A")

        # Update the active user sequence numbers and rewrite the userlog.txt file
        for i, user in enumerate(active_users, start=1):
            user[2] = i

        write_userlog()

        # Send a confirmation message to the client
        client_socket.send("Logout successful".encode())

        # Return the state information about currently logged-on users and the active user log file
        return f"Logged out: {username}. Active users: {active_users}", f"Active user log:\n{get_active_users_log()}"
    else:
        # If the user is not found in the list of active users, send an error message to the client
        client_socket.send("Error: User not found".encode())
        return "", ""

# Function to write the active user information to the userlog.txt file
def write_userlog():
    with open("userlog.txt", "w") as log_file:
        for user in active_users:
            log_file.write(f"{user[2]} {user[0]} {user[1]} {user[3]}\n")

# Function to handle joining a group, checking if the group exists and adding the user if not already a member
def handle_join_group(client_socket, username, command):
    # Split the command to extract the group name
    parts = command.split(' ')
    group_name = parts[1]

    # Check if the group exists
    if group_name not in group_chats:
        client_socket.send("The group chat does not exist.".encode())
        return

    # Retrieve information about the group
    group_info = group_chats[group_name]
    group_members = group_info["members"]

    # Check if the user is already a member of the group
    if username in group_members:
        client_socket.send(f"You are already a member of the group {group_name}.".encode())
    else:
        # Add the user to the group and notify them
        group_members.append(username)
        client_socket.send(f"You have successfully joined the group {group_name}.".encode())


# Function to handle video transfer coordination
def handle_video_transfer_request(sender_username, recipient_username):
    # Inform the recipient about the video transfer request
    recipient_socket = find_user_socket(recipient_username)
    if recipient_socket:
        recipient_socket.send(f"/p2pvideo_request {sender_username}".encode())
    else:
        print(f"Recipient {recipient_username} not found.")

# Function to find the socket of a user
def find_user_socket(username):
    for user_socket in client_sockets:
        try:
            user_socket.send(f"/check_user {username}".encode())
            response = user_socket.recv(1024).decode()
            if response == "User found":
                return user_socket
        except Exception as e:
            # Handle the case where the client socket is no longer valid
            print(f"Exception: {e}")


def handle_commands(client_socket, username):
    while True:
        command = client_socket.recv(1024).decode()

        if not command:
            break

        if command.startswith('/msgto'):
            handle_private_message(client_socket, username, command)
        elif command == '/activeuser':
            handle_active_user_request(client_socket, username)
        elif command == '/logout':
            handle_logout(client_socket, username)
            break
        elif command.startswith('/creategroup'):
            handle_group_creation(client_socket, username, command)
        elif command.startswith('/groupmsg'):
            handle_group_message(client_socket, username, command)
        elif command.startswith('/joingroup'):
            handle_join_group(client_socket, username, command)
        elif command.startswith('/p2pvideo'):
            parts = command.split(' ')
            if len(parts) == 2:
                recipient_username = parts[1]
                handle_video_transfer_request(username, recipient_username)
            else:
                print("Invalid /p2pvideo command format.")
        else:
            # Send an error message for an invalid command
            error_message = "Error.\nInvalid command! Enter one of the following commands (/msgto, /activeuser, /creategroup, /joingroup, /groupmsg, /p2pvideo, /logout):"
            client_socket.send(error_message.encode())


# Function to broadcast a message to all connected clients
def broadcast_message(message):
    to_remove = []

    for client_socket in client_sockets:
        try:
            client_socket.send(message.encode())
        except Exception as e:
            # Handle the case where the client socket is no longer valid (e.g., client disconnected)
            print(f"Exception: {e}")
            to_remove.append(client_socket)

    # Remove disconnected clients
    for sock in to_remove:
        client_sockets.remove(sock)


# Function to handle a client's connection, authentication, and command processing
def handle_client(client_socket, client_address, client_port, consecutive_attempts):
    username = authenticate_user(client_socket, consecutive_attempts)

    if username is not None:
        # Add the authenticated user to the list of active users
        with active_users_lock:
            active_users.append((username, client_address, client_port))

        # Log user information and broadcast the user's activity
        log_user_info(username, client_address, client_port)
        print(f"User {username} is now active.")
        broadcast_message(f"User {username} is now active.")

        # Add the client socket to the list and handle user commands
        client_sockets.append(client_socket)
        handle_commands(client_socket, username)

        # Remove the client socket from the list when the user logs out
        client_sockets.remove(client_socket)

        # Remove the user from the list of active users and broadcast the logout
        with active_users_lock:
            if (username, client_address, client_port) in active_users:
                active_users.remove((username, client_address, client_port))
        print(f"User {username} logged out.")
        broadcast_message(f"User {username} logged out.")
    else:
        # Close the client socket if authentication fails
        client_socket.close()

def handle_logout(client_socket, username):
    # Iterate through active_users to find and remove the user
    for user in active_users:
        if user[0] == username:
            active_users.remove(user)
            break

    # Log the user information as "N/A" for address and port
    log_user_info(username, "N/A", "N/A")
    
    # Send a farewell message to the client
    farewell_message = f"Bye, {username}!"
    client_socket.send(farewell_message.encode())

    # Return the state information about currently logged-on users and the active user log file
    return f"Logged out: {username}. Active users: {active_users}", f"Active user log:\n{get_active_users_log()}"


# Function to retrieve the active user log as a string
def get_active_users_log():
    with open("userlog.txt", "r") as log_file:
        return log_file.read()

# Main function to start the server and handle incoming client connections
def main():

        # Check if the userlog.txt file exists, and if it does, truncate it
    if os.path.exists("userlog.txt"):
        with open("userlog.txt", "w") as log_file:
            pass
    
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 4:
        print("\n===== Error usage, python3 server.py SERVER_IP SERVER_PORT NUMBER_OF_CONSECUTIVE_ATTEMPTS ======\n")
        exit(1)

    # Extract server IP, port, and consecutive attempts from command-line arguments
    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
    consecutive_attempts = sys.argv[3]

    try:
        # Convert consecutive_attempts to an integer and validate its range
        consecutive_attempts = int(consecutive_attempts)
        if not (MIN_CONSECUTIVE_ATTEMPTS <= consecutive_attempts <= MAX_CONSECUTIVE_ATTEMPTS):
            raise ValueError("Invalid value")
    except ValueError:
        # Handle invalid input for consecutive_attempts
        print(f"Invalid number of allowed failed consecutive attempts: {consecutive_attempts}. The valid value is an integer between {MIN_CONSECUTIVE_ATTEMPTS} and {MAX_CONSECUTIVE_ATTEMPTS}.")
        exit(1)
        
    # Set up the server socket and bind it to the specified address and port
    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(MAX_CONNECTIONS)

    print("\n===== Server is running =====")
    print(f"===== Waiting for connection requests from clients on {server_ip}:{server_port}...=====")
    
    # Accept and handle incoming client connections
    while True:
        client_socket, client_address = server_socket.accept()
        client_port = client_socket.getpeername()[1]
        
        # Send the number of allowed consecutive login attempts to the client
        client_socket.send(f"Allowed consecutive login attempts: {consecutive_attempts}".encode())
        
        # Create a thread to handle the client and start it
        client_handler = Thread(target=handle_client, args=(client_socket, client_address, client_port, consecutive_attempts))
        client_handler.start()


if __name__ == "__main__":
    # Call the main function when the script is executed
    main()

    # Add this code after the main function to periodically clean up blocked users
    cleanup_interval = 60  # Adjust the cleanup interval as needed
    while True:
        time.sleep(cleanup_interval)
        cleanup_blocked_users()