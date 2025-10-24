from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/test", tags=["test"])

@router.get("/{var_name}")
def execute_script(var_name: str):
    return {"message": f"This is printed {var_name}."}
