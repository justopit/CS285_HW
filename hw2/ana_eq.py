from sympy import *
x = Symbol('x', real=True)
k = Symbol('k', real=True)
one = 1 / (1-x) ** 2
two = (x + 1) / ((1-x)**3)
three = (x ** 2 + 4 * x + 1) / ((1-x) ** 4)
four = (x ** 3 + 11 * x ** 2 + 11 * x + 1 ) / ((1-x) ** 5)
print(one, two, three, four)

pg = four * (1-x) / x - 2 *  three  + two * x / (1-x) - 1 / (1-x) ** 4
pg_ans = cancel(pg)
print('pg_ans', simplify(pg_ans))


pg_2 = 1 / (x * (1-x)) * ((1-x) ** 2 * four - 2 * x * (1-x) * three + two * x ** 2) - 1 / (1-x) ** 4
print('and 2', cancel(pg_2), simplify(cancel(pg_2)))

ans = (1-x) / (4 * x) * (four + 2 * three + two) - 1 / (1-x) ** 4
print('ori ans', simplify(ans))
ans = cancel(ans)
print(simplify(ans))

plot(pg_ans, ans, (x, 0.5, 0.98), legend=True)
