from database_class import Database
from pprint import pprint



def main():
    book_database = Database()
    pprint(book_database.database)


if __name__ == "__main__":
    main() 