from utils import check_repetition

def test_duplicates():
    print("--"*50)
    print("Testing two duplicates")
    arr = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11 , 12]
    city_range = list(range(len(arr)))
    print("Test array:\n ", arr )
    print(check_repetition(city_range, arr))
    print("--"*50)

def test_non_duplicates():
    print("--"*50)
    print("Testing no duplicates")
    arr = [0, 1, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11 , 12]
    city_range = list(range(len(arr)))
    print("Test array:\n ", arr )
    print(check_repetition(city_range, arr))
    print("--"*50)

def test_more_than_two_duplicates():
    print("--"*50)
    print("Testing more than two duplicates")
    arr = [0, 1, 13, 13, 13, 13, 13, 13, 13, 13, 9, 10 , 11 , 12]
    city_range = list(range(len(arr)))
    print("Test array:\n ", arr )
    print(check_repetition(city_range, arr))
    print("--"*50)

if __name__ == "__main__": 
    test_duplicates()
    test_non_duplicates()
    test_more_than_two_duplicates()
