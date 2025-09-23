def removeDuplicates(string: list) -> list:
    final = list(set(string))
    return final


print(removeDuplicates(["a", "b", "a", "c", "b"]))