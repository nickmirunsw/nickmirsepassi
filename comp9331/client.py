# client.py

# Import necessary libraries
from socket import *
import sys
from threading import Thread
import time
import os

# UDP ====================================================

#UDP_receiver function used to receive the p2p file. 
def UDP_receiver(host,port,send_user):
    buffer=1024
    s = socket(AF_INET, SOCK_DGRAM)
    s.bind((host, port))
    addr = (host, port)
    # 1.receive the filename
    data, addr = s.recvfrom(buffer)
    print('Received File:', data.decode().strip())
    dataname = send_user + '_' + data.decode().strip()
    f = open(dataname, 'wb')

    # 2. write the content into the file
    data, addr = s.recvfrom(buffer)
    try:
        while (data):
            f.write(data)
            s.settimeout(2)
            data, addr = s.recvfrom(buffer)
    except timeout:
        f.close()
        s.close()
        print('File downloaded')


#UDP_sender function used to send the p2p file.
def UDP_sender(host,port,filename):
    buffer = 1024
    s = socket(AF_INET, SOCK_DGRAM)
    addr = (host, port)
    # sender_addr = (serverHost, udp_server_port)
    # 1.send the filenme
    s.sendto(filename.encode(), addr)
    # send the filecontent
    file = open(filename, 'rb')
    data = file.read(buffer)
    while data:
        send_bytes = s.sendto(data, addr)
        if send_bytes:
            print(f'sending a package, size = {send_bytes}......')
            data = file.read(buffer)
    s.close()
    file.close()

def initiate_p2p_video_transfer(client_socket, server_host, server_port):
    # Send the /p2pvideo command to the server
    client_socket.send("/p2pvideo".encode())
    # Receive the server's response
    response = client_socket.recv(1024).decode()
    print(response)

    # Split the response to get the sender's username and the receiver's username
    parts = response.split(' ')
    sender = parts[1]
    receiver = parts[2]

    # Create a separate thread to receive the file
    receive_thread = Thread(target=UDP_receiver, args=(server_host, server_port, sender,))
    receive_thread.start()

    # Send the file
    UDP_sender(server_host, server_port, sender + '_' + receiver + '.mp4')

    # Wait for the receive thread to finish
    receive_thread.join()

    # Delete the file after sending it
    os.remove(sender + '_' + receiver + '.mp4')

# UDP /p2pvideo ====================================================

# Function for user authentication
def authenticate_user(client_socket, allowed_attempts):
    consecutive_attempts = 0
    
    print("Please login")

    while consecutive_attempts < allowed_attempts:
        username = input("Username: ")
        client_socket.send(username.encode())
        password = input("Password: ")
        client_socket.send(password.encode())
        response = client_socket.recv(1024).decode()

        if response == "Authentication successful":
            print("Welcome to TESSENGER!")
            print("Enter one of the following commands (/msgto, /activeuser, /creategroup, /joingroup, /groupmsg, /logout, /p2pvideo):")
            return True, username
        elif "Your account is blocked" in response:
            # print(response)  # Consider uncommenting this line if you want to display the account blocked message
            return False, None
        elif consecutive_attempts < allowed_attempts - 1:
            print("Invalid Password. Please try again")
        consecutive_attempts += 1

    print("Invalid Password. Your account has been blocked. Please try again later.")
    return False, None


# Function to send a private message
def handle_private_message(sender_socket, sender, recipient, message):
    sender_socket.send(f"/msgto {recipient} {message}".encode())
    response = sender_socket.recv(1024).decode()
    print(response)

# Function to request active user data
def handle_active_user_request(client_socket):
    client_socket.send("/activeuser".encode())
    active_user_data = client_socket.recv(1024).decode()
    print(active_user_data)


# Function to handle user logout
def handle_logout(client_socket):
    # Send logout command to the server
    client_socket.send("/logout".encode())
    # Receive and print the server's response
    response = client_socket.recv(1024).decode()
    print(response)
    # Close the client socket
    client_socket.close()


# Function for creating a group chat
def handle_group_creation(client_socket):
    # Get group name and members' usernames from user input
    group_name = input("Enter group name: ")
    members = input("Enter usernames separated by space: ").split()
    # Send create group command to the server
    client_socket.send(f"/creategroup {group_name} {' '.join(members)}".encode())
    # Receive and print the server's response
    response = client_socket.recv(1024).decode()
    print(response)


# Function for sending a message to a group chat
def handle_group_message(client_socket):
    # Get group name and message from user input
    group_name = input("Enter group name: ")
    message = input("Enter your message: ")
    # Send group message command to the server
    client_socket.send(f"/groupmsg {group_name} {message}".encode())
    # Receive and print the server's response
    response = client_socket.recv(1024).decode()
    print(response)


# Function to display available commands
def display_available_commands():
    print("Available commands:")
    print("/msgto USERNAME MESSAGE_CONTENT - Send a private message")
    print("/activeuser - Display active users")
    print("/logout - Log out")
    print("/creategroup - Create a group chat")
    print("/groupmsg - Send a message to a group chat")
    print("/joingroup GROUP_NAME - Request to join a group")
    print("/p2pvideo - Initiate a peer-to-peer video transfer")


# Function to handle server responses in a separate thread
def handle_server_responses(client_socket):
    # Continuously receive and print server responses
    while True:
        response = client_socket.recv(1024).decode()
        if not response:
            break
        print(response)


def handle_join_group(client_socket, command):
    # Split the command to get the group name
    parts = command.split(' ')
    if len(parts) != 2:
        # Print an error message for invalid command format
        print("Invalid command format. Please use '/joingroup GROUP_NAME'.")
        return

    # Get the group name from the command
    group_name = parts[1]
    # Send join group command to the server
    client_socket.send(f"/joingroup {group_name}".encode())
    # Receive and print the server's response
    response = client_socket.recv(1024).decode()
    print(response)


def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("\n===== Error usage, python3 client.py SERVER_IP SERVER_PORT ======\n")
        exit(1)

    # Extract server IP and port from command-line arguments
    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    server_address = (server_host, server_port)

    # Create a socket and connect to the server
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect(server_address)

    # Receive the number of allowed consecutive login attempts from the server
    allowed_attempts_message = client_socket.recv(1024).decode()
    allowed_attempts = int(allowed_attempts_message.split(': ')[1])

    # Authenticate the user and receive the username
    authenticated, username = authenticate_user(client_socket, allowed_attempts)
    
    if authenticated:
        # Create a separate thread to handle server responses
        response_thread = Thread(target=handle_server_responses, args=(client_socket,))
        response_thread.start()

        while True:
            command = input(" ")
            
            # If the user types '/logout', wait for the response thread to finish and exit the loop
            if command == '/logout':
                # Send the user's command to the server
                client_socket.send(command.encode())
                # Wait for the response thread to finish with a timeout
                response_thread.join(timeout=1)
                # Close the client socket after the user logs out
                client_socket.close()
                break
            
            elif command == '/p2pvideo':
                # Call the function to handle the entire process of initiating the file transfer
                initiate_p2p_video_transfer(client_socket, server_host, server_port)

            else:
                # Send the user's command to the server for other commands
                client_socket.send(command.encode())


        # Ensure the response thread is terminated if it hasn't finished yet
        if response_thread.is_alive():
            response_thread.join()
    else:
        # Inform the user that their account is blocked
        print("Your account is blocked due to multiple login failures. Please try again later.")
        
if __name__ == "__main__":
    main()
