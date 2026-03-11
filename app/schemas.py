from pydantic import BaseModel, Field

class CarFeatures(BaseModel):
    """
    Pydantic schema representing the required features to predict the price of a car.
    These features strictly mirror the dataset features expected by the training pipeline.
    """
    Year: float = Field(..., gt=1900, description="The manufacturing year of the continuous vehicle.")
    Mileage: float = Field(..., ge=0, description="Total mileage driven.")
    City: str = Field(..., description="The city where the car is located.")
    State: str = Field(..., min_length=2, max_length=2, description="The 2-letter state code.")
    Make: str = Field(..., description="Brand of the car.")
    Model: str = Field(..., description="Model string designation of the vehicle.")
