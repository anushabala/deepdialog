__author__ = 'anushabala'
from socketIO_client import SocketIO, BaseNamespace
import time


def print_intro():
    print "hello there"


def process_message(data):
    print "processing something", data
    time.sleep(2)
    print "done"


class ChatNamespace(BaseNamespace):

    def on_connect(self):
        print_intro()

    def on_message(self, data):
        process_message(data)

if __name__=="__main__":
    socket = SocketIO('localhost', 5000, ChatNamespace)
    chat = socket.define(ChatNamespace, '/bot')
    print "hi?"
    print "hey?"
    # socket.on('connect', print_intro())
    # socket.on('message', process_message)
    socket.wait()






