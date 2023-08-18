from mpspdz.ExternalIO.client import *
from mpspdz.ExternalIO.domains import *

def client(client_id, n_parties, port, param_num, workers, chunk_size, data, precision=12):
    """
    Acts as a client for the mpc computation parties. It secret-shares the inputs for the computation parties
    and receive the output of the computation.
    client_id: id given to computation parties
    n_parties: number of computation parties
    port: port computation parties are listing on
    param_num: number of parameters per gradient
    workers: number of workers
    chunk_size: amount of values submitted at one time
    data: client input for computation
    precision: bit precision for fixed-point numbers
    """

    client = Client(['localhost'] * n_parties, port, client_id)

    type = client.specification.get_int(4)
    if type == ord('R'):
        domain = Z2(client.specification.get_int(4))
    elif type == ord('p'):
        domain = Fp(client.specification.get_bigint())
    else:
        raise Exception('invalid type')

    # send data
    input = [domain(d * 2 ** precision) for d in data]
    full_batches = (param_num * workers) // chunk_size

    for i in range(full_batches):
        client.send_private_inputs(input[i * chunk_size:(i + 1)*chunk_size])
    client.send_private_inputs(input[full_batches * chunk_size:])

    print("Data send")

    # receive data
    full_batches = param_num // chunk_size
    rest = param_num - full_batches * chunk_size

    output = []

    for i in range(full_batches):
        output += client.receive_outputs(domain, chunk_size)
    output += client.receive_outputs(domain, rest)
    output = [((x.v - 2**64) / 2**precision if x.v >= 2**63 else x.v / 2**precision) for x in output]   # convert 2's complement integers to float

    return output