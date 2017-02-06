"""
Interface to help interact with the neural networks.
"""
import multiprocessing


class Interface:
    """
    A class to help interact with the neural networks.
    """

    def __init__(self, network_class):
        self.queue = multiprocessing.Queue()
        self.network = network_class(message_queue=self.queue)

    def run(self):
        """
        Runs the interface between the user and the network.
        """
        # Run the network.
        if self.network.settings.run_mode == 'test':
            self.test()
        else:
            self.train()

        print('Program done.')

    def train(self):
        """
        Runs the main interactions between the user and the network during training.
        """
        self.network.start()
        while True:
            user_input = input()
            if user_input == 's':
                print('Save requested.')
                self.queue.put('save')
            elif user_input == 'q':
                print('Quit requested.')
                self.queue.put('quit')
                self.network.join()
                break

    def test(self):
        """
        Runs the network prediction.
        """
        self.network.test()
