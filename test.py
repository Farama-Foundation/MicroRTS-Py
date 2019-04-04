import socket

s = socket.socket()
s.bind(('', 9898))
s.listen(5)

# a forever loop until we interrupt it or  
# an error occurs 

conn, addr = s.accept()
print('Got connection from', addr)

conn.send(('%s\n' % "").encode('utf-8'))
print(conn.recv(4096).decode('utf-8'))

conn.send(('%s\n' % "").encode('utf-8'))
print(conn.recv(4096).decode('utf-8'))