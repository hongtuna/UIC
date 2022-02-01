import socket
import time
import logging

# msgFromClient       = "Hello UDP Server"
# bytesToSend         = str.encode(msgFromClient)
serverAddressPort   = ("192.168.1.45", 20001)
bufferSize          = 1024
 
# Create a UDP socket at client side
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

i = 0
# Send to server using created UDP socket
while i < 100:
    i += 1
    MESSAGE = str(i).encode('utf-8')
    print('sending parameter',i,'to client')
    UDPClientSocket.sendto(MESSAGE, serverAddressPort)
    time.sleep(1)
    
logging.warning('closing the socket')
UDPClientSocket.close()

# msgFromServer = UDPClientSocket.recvfrom(bufferSize)
# msg = "Message from Server {}".format(msgFromServer[0])
# print(msg)

