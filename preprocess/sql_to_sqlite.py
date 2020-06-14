#!/usr/bin/python3

import argparse
import sqlite3
import os


def change_extension(filename, new_extension):
    last_dot_index = filename.rfind('.')
    if last_dot_index == -1:
        raise ValueError("Please provide output file name using `-o` or `--output` argument")
    filename = filename[:filename.rfind('.')] + new_extension
    return filename


def get_filenames(args):
    input_filename = args.input
    output_filename = args.output
    if output_filename is None:
        output_filename = change_extension(input_filename, '.sqlite')
    return input_filename, output_filename


def convert_sql_to_sqlite(input_filename, output_filename, encoding='utf-8'):
    # Delete file so the database would be clean
    if os.path.exists(output_filename):
        os.remove(output_filename)

    try:
        connection = sqlite3.connect(output_filename)
        cursor = connection.cursor()
        with open(input_filename, 'r', encoding=encoding) as f:
            sql = "".join(f.readlines())

        cursor.executescript(sql)
        connection.commit()
    except sqlite3.Error as e:
        print('SQLite3 error occurred:')
        print(e)
        raise
    finally:
        if connection:
            connection.close()


def main(args):
    input_filename, output_filename = get_filenames(args)
    encoding = args.encoding
    convert_sql_to_sqlite(input_filename, output_filename, encoding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script can convert .sql to .sqlite format")
    parser.add_argument('input', type=str, help='input filename (.sql format)')
    parser.add_argument('-o', '--output', type=str, help='output filename (.sqlite format)')
    parser.add_argument('-e', '--encoding', type=str, help='encoding of the input filename (default: utf-8)', default='utf-8')
    args = parser.parse_args()

    main(args)
