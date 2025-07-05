import numpy as np

nInst = 50
currentPos = np.zeros(nInst, dtype=int)       # make these integer from the start

def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    assert nins == nInst, f"Expected {nInst} instruments, got {nins}"

    k = 50  # use k-day returns

    # Need at least k+1 days to compute a k-day return
    if nt < k + 1:
        return currentPos

    # Compute k-day log-returns
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -1 - k])

    # Compute norm and guard against zero
    lNorm = np.linalg.norm(lastRet)
    if lNorm == 0:
        return currentPos
    lastRet /= lNorm

    # Generate position changes (sell positive signals, buy negative)
    scale = 1400
    rpos = np.array([
        int(-scale * ret_i / prcSoFar[i, -1])
        for i, ret_i in enumerate(lastRet)
    ], dtype=int)

    # Update and return integer positions
    currentPos = currentPos + rpos
    return currentPos
