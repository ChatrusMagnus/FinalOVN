import random
import numpy as np
from FinalOVN.Network import Network
from FinalOVN.Connection import Connection
import matplotlib.pyplot as plt

def main():
    network = Network('nodes.json')
    network.connect()
    network.draw()
    node_labels = list(network.nodes.keys())
    connections = []
    for i in range(100):
        random.shuffle(node_labels)
        connection = Connection(node_labels[0], node_labels[-1])
        connections.append(connection)
    streamed_connections = network.stream(connections, best='snr')
    snrs = [connection.snr for connection in streamed_connections]
    plt.hist(snrs, bins=10)
    plt.title('SNR Distribution ')
    plt.show()
    df = network.route_space
    df.to_html('free.html')
    rbs = [connection.calculate_capacity() for connection in streamed_connections]
    plt.hist(rbs, bins=10)
    plt.title('Bitrate Distribution [Gbps]')
    plt.show()

    print('Total Capacity: {:.2f} Tbps '.format(np.sum(rbs) * 1e-3))
    print('Avg Capacity: {:.2f} Gbps '.format(np.mean(rbs)))

if __name__ == "__main__":
    main()