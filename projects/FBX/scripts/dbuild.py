import os
import datetime
import argparse
import requests
import threading


DEFAULT_HOST = "192.168.3.15:5000"


def send(host, comm):
    _addr = 'http://{}/{}'.format(host, comm)
    print("Send Get", _addr)
    r = requests.get(_addr)
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"), "Response", r.status_code, "Status", r.text)
    return r.text


def run1(host, comm, interval):
    threading.Timer(interval, run1, args=(host, comm, interval)).start()
    send(host, comm)


def run2(host, comm, interval):
    threading.Timer(interval, run2, args=(host, comm, interval)).start()
    r = send(host, comm)
    if int(r) == 1:
        send(host, "start")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host Address, e.g. {}".format(DEFAULT_HOST),
                        type=str, default=DEFAULT_HOST)
    args = parser.parse_args()
    print("Args: Host", args.host)

    run1(args.host, "start", 86400)
    run2(args.host, "check?mode=1", 60)
    print("Run")


if __name__ == '__main__':
    main()
