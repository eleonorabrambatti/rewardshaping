from pydantic import BaseModel, Field, root_validator
 
 
class SupplyChainBaseSchema(BaseModel):
    """Base schema for the supply chain."""
 
    class Config:
        """Pydantic configuration."""
 
        orm_mode = True
        use_enum_values = True
 
 
class ItemSchema(SupplyChainBaseSchema):
    """Schema for the items involved in the supply chain."""
 
    name: str = Field(default="")
    lead_time: int = Field(ge=0)
    expiration_time: int = Field(ge=0)
    selling_price: float = Field(ge=0.0)
    order_cost: dict[int, float]
    penalty_cost: float = Field(ge=0.0)
    expiration_cost: float = Field(ge=0.0)
 
 
class StockSchema(SupplyChainBaseSchema):
    """Schema for the stocks involved in the supply chain."""
 
    name: str = Field(default="")
    item_schema: ItemSchema
    capacity: int = Field(ge=0)
    storage_cost: float = Field(ge=0.0)
    lost_items_probability: float = Field(default=0.0, ge=0, le=100)
 
 
class SupplyChainSchema(SupplyChainBaseSchema):
    """Schema for the supply chain."""
 
    max_time: int = Field(ge=0)
    len_demand_history: int = Field(ge=0)
    len_forecast_history: int = Field(ge=0)
    stock_schema: StockSchema
    max_orders: int = Field(ge=0)
    items_per_order: int = Field(ge=0)
    demand_fn: str
    demand_fn_args: dict
    forecast_fn: str
    forecast_fn_args: dict
    initial_stock: list[int]
    initial_in_transit: list[int]
    scalar_state: bool
 
    @root_validator(skip_on_failure=True)
    def initial_stock_validator(cls, values):
        initial_stock = values.get("initial_stock")
        stock_schema = values.get("stock_schema")
        if sum(initial_stock) > stock_schema.capacity:
            raise ValueError("Initial stock must be less than or equal to capacity")
        if len(initial_stock) != stock_schema.item_schema.expiration_time:
            raise ValueError(
                "Initial stock must have the same length as the expiration time"
            )
        return values
 
    @root_validator(skip_on_failure=True)
    def history_validator(cls, values):
        scalar_state = values.get("scalar_state")
        if scalar_state:
            values["len_demand_history"] = 0
            values["len_forecast_history"] = 0
        return values