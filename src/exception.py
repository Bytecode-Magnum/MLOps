import sys
import logging

#: this function take the eror which is raise using raise CustomException(e,sys), and we design a custom exception message
def error_message_details(error,error_detail:sys):
  _,_,exc_tb=error_detail.exc_info()
  file_name=exc_tb.tb_frame.f_code.co_filename  #: file name in which the error is occured
  error_message="error occured in python script named [{0}] line number [{1}] error message  [{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)
  )
  return error_message

class CustomException(Exception):
  def __init__(self,error_message,error_detail:sys) :   #: CustomException(e,sys)
    super().__init__(error_message)
    self.error_message=error_message_details(error_message,error_detail=error_detail) 
     #: formatted the error  message using error_message_details function

  def __str__(self):
    return self.error_message

if __name__=="__main__":
  try:
    a=1/0
  except Exception as e:
    logging.info("Divide by zero error")
    raise CustomException(e,sys)