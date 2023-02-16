
import modelo as m

def clasificar(mensaje):
      
    Pspammensaje = m.probspam
    Phammensaje = m.probham

    for palabra in mensaje.split():
        if palabra in m.psw:
            Pspammensaje += (Pspammensaje * m.psw[palabra])
          
        if palabra in m.phw:
            Phammensaje += (Phammensaje * m.phw[palabra])

    print('P(Spam|mensaje):', Pspammensaje)
    print('P(Ham|mensaje):', Phammensaje)

    if Phammensaje > Pspammensaje:
        print('Este mensaje es muy probable que sea ham')
    elif Phammensaje < Pspammensaje:
        print('Este mensaje es muy probable que sea spam')
    else:
        print('Equal proabilities, have a human classify this!')

def testear(mensaje):
    Pspammensaje =m.probspam
    Phammensaje = m.probham

    for palabra in mensaje.split():
        if palabra in m.psw:
            Pspammensaje += (Pspammensaje * m.psw[palabra])
          
        if palabra in m.phw:
            Phammensaje += (Phammensaje *m.phw[palabra])
    
    if Phammensaje > Pspammensaje:
       return 'ham'
    elif Pspammensaje > Phammensaje:
       return 'spam'
    else:
       return 'no se sabe'