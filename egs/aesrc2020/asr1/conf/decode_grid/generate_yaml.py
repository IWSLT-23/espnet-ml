idx=1
start=idx
ctc=[0,0.25,.5]
pen=[0, .1, .2, 0.3]
for c in ctc:
    for p in pen:
            params = "batchsize: 0\nbeam-size: 10\npenalty: "+str(p)+"\nmaxlenratio: 0.0\nminlenratio: 0.0\nctc-weight: " + str(c)
            with open("decode" + str(idx) + ".yaml", "w") as f:
                f.write(params)

            idx += 1
    

print "Wrote ", idx-start, " yaml file(s)"
