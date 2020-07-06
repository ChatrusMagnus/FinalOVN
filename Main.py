from FinalOVN.Analytics import Monte_carlo

def main():
    simulation_number=5
    channel=10
    request_rate=600
    best='snr'
    transceiver='shannon'
    multiplier=1
    analitic=Monte_carlo(simulation_number,channel,request_rate,best,transceiver,multiplier,'LEAF')
    analitic.run_simulations('')

if __name__ == "__main__":
    main()