def monthly_payment(principal, years, rate):
    """
    Calculate the monthly payment of a fixed term mortgage
    """
    monthly_rate = rate / 100 / 12
    months = years * 12
    payment = monthly_rate * principal / (1 - (1 + monthly_rate) ** (-months))
    return payment

def calculate():
    principle = float(input(">>>principal: "))
    years = float(input(">>>years: "))
    rate = float(input(">>>rate: "))
    payment = monthly_payment(principle, years, rate)
    print("Monthly payment: %f" % payment)

def main():
    while True:
        calculate()

if __name__ == '__main__':
    main()