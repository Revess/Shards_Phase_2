# Print iterations progress // made by: https://stackoverflow.com/a/34325723

def printProgressBar (iteration, total, gen_loss=0,disc_loss=0,acc=0, title="progress bar", prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if gen_loss == 0 and acc == 0 and disc_loss == 0 :
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    else:
        print(f'\r{title} >> {prefix} |{bar}| {percent}% {suffix} gen_loss: {"{:.6f}".format(gen_loss)} disc_loss: {"{:.6f}".format(disc_loss)} accuracy: {"{:.2f}".format(acc)}', end = printEnd)

    # Print New Line on Complete
    if iteration == total: 
        print()