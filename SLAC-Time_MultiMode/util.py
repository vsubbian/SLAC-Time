
def inv_list(l):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i
    return d


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


