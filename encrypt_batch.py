#!/usr/bin/env python3
"""
encrypt_batch.py
================
Encrypts a review batch (.json.gz) with a password, for a hosted alignment review page (?review=NAME reads
reviews/NAME.enc). Format: b"ALN1", iterations (4 bytes, big-endian), 16-byte salt, 12-byte IV, then the
AES-256-GCM ciphertext of the .json.gz bytes; the key is PBKDF2-SHA256(password, salt, iterations).
The viewer decrypts it in the browser (Core.decryptBatchBytes); nothing else is needed on the server.

    python encrypt_batch.py align_review/align01.json.gz reviews/align01.enc --password-file PASSWORD.txt

Without --password-file a random password is made and written to --new-password-file (never to the repository).
Anyone with the .enc file can try passwords offline, so use a long random one.
"""
import argparse
import os
import secrets
import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITER = 600_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('out')
    ap.add_argument('--password-file', help='file holding the password (first line)')
    ap.add_argument('--new-password-file', help='write a new random password here (outside the repository)')
    a = ap.parse_args()
    if a.password_file:
        pw = open(a.password_file).readline().strip()
    elif a.new_password_file:
        pw = secrets.token_urlsafe(18)
        fd = os.open(a.new_password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(pw + '\n')
    else:
        raise SystemExit('give --password-file or --new-password-file')
    data = open(a.batch, 'rb').read()
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(pw.encode())
    ct = AESGCM(key).encrypt(iv, data, None)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'wb') as f:
        f.write(b'ALN1' + struct.pack('>I', ITER) + salt + iv + ct)
    print(f'wrote {a.out} ({len(ct) / 1e6:.1f} MB) from {a.batch}')


if __name__ == '__main__':
    main()
