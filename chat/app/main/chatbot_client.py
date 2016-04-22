__author__ = 'anushabala'
from socketIO_client import SocketIO, BaseNamespace

socket = SocketIO('localhost', 5000)
chat = socket.define(BaseNamespace, '/chat')
chat.emit('text', 'hello chat my name is Anderson')
