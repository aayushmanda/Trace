from model import *
for n in (4,8,10):
    out,st=data(n,.25)
    _,_,a,mm=loss_grad(outcome_executor(n),out); print(n,'outcome executor acc',a,'min margin',mm)
    _,_,a,mm=loss_grad(step_executor(n,6*(n+1)),st); print(n,'step executor acc',a,'min margin',mm)
