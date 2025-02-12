with open('l1.data') as fin:
    data = fin.readlines()

values = [int(f.split(',')[1]) for f in data]

print (','.join([str(f) for f in values]))