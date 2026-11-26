from field_generation_v1.training import main,status,PauseRequested

if __name__=='__main__':
    try:main()
    except PauseRequested as error:status(phase='paused',message=str(error))
    except Exception as error:
        status(phase='failed',error=repr(error));raise
