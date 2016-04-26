__author__ = 'anushabala'
from socketIO_client import SocketIO, BaseNamespace

socket = SocketIO('localhost', 5000,
                  params={'bot':1},
                  cookies={'sid':'BOT'})
chat = socket.define(BaseNamespace, '/main')
chat.emit('text', 'hello chat my name is Anderson')
