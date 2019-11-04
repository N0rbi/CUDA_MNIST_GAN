#include <iostream>
#include <typeinfo>

int main() {
    std::cout << typeid("asd").name() << std::endl;
    return 0;
}